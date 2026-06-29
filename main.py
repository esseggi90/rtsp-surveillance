"""
RTSP Surveillance Web – Backend FastAPI
Legge stream RTSP, fa motion detection + YOLO, serve:
  - MJPEG stream per ogni camera
  - SSE per eventi live (motion, persone)
  - API REST per configurazione
  - Dashboard HTML

Configurazione via env var CAMERAS (JSON) o config.json
"""

import os, json, time, threading, asyncio, logging
from datetime import datetime
from typing import AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("surveillance")

app = FastAPI(title="RTSP Surveillance")

# ── Percorsi template/static ───────────────────────────────────────────
BASE = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE, "templates"))
static_path = os.path.join(BASE, "static")
if os.path.isdir(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    # 1. env var CAMERAS → JSON array completo (override totale)
    env = os.environ.get("CAMERAS")
    if env:
        try:
            return {"cameras": json.loads(env)}
        except Exception:
            pass

    # 2. config.json locale con sostituzione RTSP_HOST e RTSP_PASSWORD
    cfg_path = os.path.join(BASE, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            raw = f.read()
        # Sostituisce i placeholder con le env var
        host     = os.environ.get("RTSP_HOST",     "192.168.1.2")
        password = os.environ.get("RTSP_PASSWORD", "88888888")
        raw = raw.replace("RTSP_HOST", host).replace("RTSP_PASSWORD", password)
        return json.loads(raw)

    # 3. default demo
    return {"cameras": [
        {"id": 1, "name": "Demo",
         "url": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
         "enabled": True, "motion_zones": []}
    ]}

CONFIG = load_config()

# ══════════════════════════════════════════════════════════════════════
#  Stato globale per camera
# ══════════════════════════════════════════════════════════════════════

class CameraState:
    def __init__(self, cam: dict):
        self.id      = cam["id"]
        self.name    = cam["name"]
        self.url     = cam["url"]
        self.zones   = cam.get("motion_zones", [])
        self.enabled = cam.get("enabled", True)

        self.frame_lock  = threading.Lock()
        self.last_frame: bytes | None = None          # JPEG bytes
        self.last_frame_raw: np.ndarray | None = None # BGR numpy

        self.events: list[dict] = []           # ultimi 200 eventi
        self.event_lock = threading.Lock()

        self._sse_queues: list[asyncio.Queue] = []
        self._sse_lock = threading.Lock()

        self.connected = False

    def push_event(self, level: str, msg: str):
        ev = {
            "time":   datetime.now().strftime("%H:%M:%S"),
            "cam":    self.name,
            "level":  level,
            "msg":    msg,
        }
        with self.event_lock:
            self.events.append(ev)
            if len(self.events) > 200:
                self.events.pop(0)
        # Notifica tutti i listener SSE
        with self._sse_lock:
            for q in self._sse_queues:
                try:
                    q.put_nowait(ev)
                except Exception:
                    pass

    def add_sse_queue(self, q: asyncio.Queue):
        with self._sse_lock:
            self._sse_queues.append(q)

    def remove_sse_queue(self, q: asyncio.Queue):
        with self._sse_lock:
            try:
                self._sse_queues.remove(q)
            except ValueError:
                pass


CAMERAS: dict[int, CameraState] = {}

# ══════════════════════════════════════════════════════════════════════
#  YOLO (lazy singleton)
# ══════════════════════════════════════════════════════════════════════

_yolo_model = None
_yolo_lock  = threading.Lock()
_yolo_ready = False

def get_yolo():
    global _yolo_model, _yolo_ready
    with _yolo_lock:
        if _yolo_model is None:
            try:
                from ultralytics import YOLO
                import logging as _l
                _l.getLogger("ultralytics").setLevel(_l.WARNING)
                _yolo_model = YOLO("yolov8n.pt")
                # warm-up
                dummy = np.zeros((240, 320, 3), dtype=np.uint8)
                _yolo_model(dummy, verbose=False, classes=[0])
                _yolo_ready = True
                log.info("YOLOv8 pronto")
            except Exception as e:
                log.warning(f"YOLO non disponibile: {e}")
        return _yolo_model if _yolo_ready else None

def yolo_detect(frame: np.ndarray, conf: float = 0.25) -> list[dict]:
    model = get_yolo()
    if model is None:
        return []
    h, w = frame.shape[:2]
    scale = min(640/w, 640/h, 1.0)
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
    with _yolo_lock:
        results = model(small, verbose=False, classes=[0], conf=conf)
    out = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            out.append({
                "bbox": [int(x1/scale), int(y1/scale),
                         int(x2/scale), int(y2/scale)],
                "conf": float(box.conf[0])
            })
    return out


# ══════════════════════════════════════════════════════════════════════
#  Worker per ogni camera
# ══════════════════════════════════════════════════════════════════════

def camera_worker(state: CameraState):
    """Thread dedicato per una telecamera."""
    retry_delay = 5
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=False)
    zone_last_alert: dict[int, float] = {}

    while True:
        state.connected = False
        state.push_event("info", "Connessione in corso...")

        # Apri capture
        if state.url.isdigit():
            cap = cv2.VideoCapture(int(state.url))
        elif os.path.isfile(state.url):
            cap = cv2.VideoCapture(state.url)
        else:
            # Forza TCP per stream RTSP remoti (più affidabile di UDP)
            cap = cv2.VideoCapture(state.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            # Imposta trasporto TCP via opzioni FFMPEG
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        # Attendi connessione
        connected = False
        for _ in range(16):
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    connected = True
                    break
            time.sleep(0.5)

        if not connected:
            cap.release()
            state.push_event("error", "Connessione fallita, riprovo...")
            time.sleep(retry_delay)
            continue

        state.connected = True
        state.push_event("info", "Connesso")
        log.info(f"[Cam {state.id}] Connessa: {state.url}")

        is_file = os.path.isfile(state.url)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = 1.0 / src_fps if is_file else 0
        last_t = time.time()
        frame_n = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    bg_sub = cv2.createBackgroundSubtractorMOG2(
                        history=500, varThreshold=25, detectShadows=False)
                    last_t = time.time()
                    continue
                break

            frame_n += 1
            h, w = frame.shape[:2]

            # ── Motion detection per zona ──────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            person_boxes: list[dict] = []

            for zi, zone in enumerate(state.zones):
                if not zone.get("enabled", True):
                    continue
                pts = zone.get("points", [])
                if len(pts) < 3:
                    continue

                min_area  = zone.get("min_area", 500)
                cooldown  = zone.get("cooldown", 10)
                use_yolo  = zone.get("detect_persons", False)
                yolo_conf = zone.get("person_conf", 25) / 100.0
                actions   = zone.get("actions", {})

                zmask = np.zeros((h, w), dtype=np.uint8)
                poly  = np.array([[int(p[0]*w), int(p[1]*h)] for p in pts],
                                 dtype=np.int32)
                cv2.fillPoly(zmask, [poly], 255)

                gm   = cv2.bitwise_and(gray, gray, mask=zmask)
                diff = bg_sub.apply(gm)
                diff = cv2.dilate(diff, None, iterations=2)
                diff = cv2.bitwise_and(diff, diff, mask=zmask)

                cnts, _ = cv2.findContours(
                    diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                max_area = max((cv2.contourArea(c) for c in cnts), default=0)

                if max_area < min_area:
                    continue

                # Filtro YOLO opzionale
                if use_yolo:
                    boxes = yolo_detect(frame, yolo_conf)
                    # Verifica che almeno una box sia nella zona
                    in_zone = []
                    for b in boxes:
                        cx_n = ((b["bbox"][0]+b["bbox"][2])/2) / w
                        cy_n = ((b["bbox"][1]+b["bbox"][3])/2) / h
                        poly_n = np.array([[p[0],p[1]] for p in pts],
                                          dtype=np.float32)
                        if cv2.pointPolygonTest(
                                poly_n, (float(cx_n), float(cy_n)), False) >= 0:
                            in_zone.append(b)
                    if not in_zone:
                        continue
                    person_boxes.extend(in_zone)

                # Cooldown
                now  = time.time()
                last = zone_last_alert.get(zi, 0)
                if now - last < cooldown:
                    continue
                zone_last_alert[zi] = now

                zname  = zone.get("name", f"Zona {zi+1}")
                prefix = "🚶 " if use_yolo else "🔴 "
                state.push_event("motion", f"{prefix}[{zname}] Movimento rilevato")

                # Azioni
                if actions.get("snapshot"):
                    _save_snapshot(frame, state.id, zname)
                if actions.get("beep"):
                    pass  # non disponibile server-side
                if actions.get("log"):
                    pass  # già loggato con push_event

            # ── Disegna overlay sul frame ──────────────────────────────
            vis = frame.copy()
            for zi, zone in enumerate(state.zones):
                pts = zone.get("points", [])
                if len(pts) >= 3:
                    poly = np.array([[int(p[0]*w), int(p[1]*h)] for p in pts],
                                    dtype=np.int32)
                    color = (0, 200, 100) if zone.get("enabled", True) else (80, 80, 80)
                    overlay = vis.copy()
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                    cv2.polylines(vis, [poly], True, color, 2)
                    cx = int(np.mean([p[0] for p in pts]) * w)
                    cy = int(np.mean([p[1] for p in pts]) * h)
                    cv2.putText(vis, zone.get("name", f"Z{zi+1}"),
                                (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255,255,255), 2)

            for b in person_boxes:
                x1,y1,x2,y2 = b["bbox"]
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,80), 2)
                cv2.putText(vis, f"{b['conf']:.0%}",
                            (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,80), 2)

            # ── Encode JPEG ────────────────────────────────────────────
            _, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with state.frame_lock:
                state.last_frame     = jpg.tobytes()
                state.last_frame_raw = frame.copy()

            # ── Throttle file locale ───────────────────────────────────
            if is_file and interval > 0:
                next_t = last_t + interval
                now    = time.time()
                if next_t > now:
                    time.sleep(next_t - now)
                last_t = max(next_t, time.time())

        cap.release()
        state.push_event("error", "Stream perso, riconnessione...")
        time.sleep(retry_delay)


def _save_snapshot(frame: np.ndarray, cam_id: int, zone_name: str):
    snap_dir = os.path.join(BASE, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(snap_dir, f"cam{cam_id}_{zone_name}_{ts}.jpg")
    cv2.imwrite(path, frame)


# ══════════════════════════════════════════════════════════════════════
#  Startup / Shutdown
# ══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    # Avvia thread per ogni camera abilitata
    for cam in CONFIG.get("cameras", []):
        if not cam.get("enabled", True):
            continue
        state = CameraState(cam)
        CAMERAS[cam["id"]] = state
        t = threading.Thread(target=camera_worker, args=(state,), daemon=True)
        t.start()
    # Carica YOLO in background
    threading.Thread(target=get_yolo, daemon=True).start()
    log.info(f"Avviate {len(CAMERAS)} telecamere")


# ══════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    cams = [{"id": s.id, "name": s.name} for s in CAMERAS.values()]
    return templates.TemplateResponse(
        "index.html", {"request": request, "cameras": cams})


@app.get("/stream/{cam_id}")
async def video_stream(cam_id: int):
    """MJPEG stream per una singola camera."""
    state = CAMERAS.get(cam_id)
    if not state:
        return StreamingResponse(iter([]), media_type="text/plain")

    async def generate():
        while True:
            with state.frame_lock:
                jpg = state.last_frame
            if jpg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(0.04)  # ~25fps max

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/events")
async def sse_events(request: Request):
    """Server-Sent Events: eventi da tutte le telecamere."""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    for state in CAMERAS.values():
        state.add_sse_queue(q)

    async def generate() -> AsyncGenerator[str, None]:
        # Manda gli ultimi 20 eventi storici
        history = []
        for state in CAMERAS.values():
            with state.event_lock:
                history.extend(state.events[-5:])
        history.sort(key=lambda e: e["time"])
        for ev in history:
            yield f"data: {json.dumps(ev)}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            for state in CAMERAS.values():
                state.remove_sse_queue(q)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/cameras")
async def api_cameras():
    return [
        {
            "id":        s.id,
            "name":      s.name,
            "connected": s.connected,
            "zones":     len(s.zones),
        }
        for s in CAMERAS.values()
    ]


@app.get("/api/events")
async def api_events(limit: int = 50):
    all_events = []
    for state in CAMERAS.values():
        with state.event_lock:
            all_events.extend(state.events)
    all_events.sort(key=lambda e: e["time"], reverse=True)
    return all_events[:limit]


@app.get("/api/config")
async def api_config():
    return CONFIG


@app.post("/api/config")
async def api_update_config(request: Request):
    """Aggiorna configurazione (telecamere + zone) a runtime."""
    global CONFIG
    body = await request.json()
    CONFIG = body
    # Salva su disco
    cfg_path = os.path.join(BASE, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    return {"status": "ok", "message": "Riavvia il server per applicare le modifiche alle telecamere"}
