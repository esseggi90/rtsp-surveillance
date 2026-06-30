"""
RTSP Surveillance Web – FastAPI backend
Env vars:
  RTSP_HOST, RTSP_PASSWORD       → credenziali NVR
  STREAM_FPS   (default 10)      → fps MJPEG verso browser
  PROCESS_FPS  (default 6)       → fps analisi motion/YOLO
  JPEG_QUALITY (default 60)      → qualità JPEG
  TELEGRAM_BOT_TOKEN             → token bot Telegram (globale)
  TELEGRAM_CHAT_ID               → chat_id Telegram (globale)
"""

import os, json, time, threading, asyncio, logging
from datetime import datetime
from typing import AsyncGenerator

import cv2
import numpy as np
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("surveillance")

app  = FastAPI(title="RTSP Surveillance")
BASE = os.path.dirname(__file__)
templates  = Jinja2Templates(directory=os.path.join(BASE, "templates"))
static_dir = os.path.join(BASE, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── Env ───────────────────────────────────────────────────────────────
STREAM_FPS         = int(os.environ.get("STREAM_FPS",    "10"))
PROCESS_FPS        = int(os.environ.get("PROCESS_FPS",   "6"))
JPEG_QUALITY       = int(os.environ.get("JPEG_QUALITY",  "60"))
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    env = os.environ.get("CAMERAS")
    if env:
        try:
            return {"cameras": json.loads(env)}
        except Exception:
            pass
    cfg = os.path.join(BASE, "config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            raw = f.read()
        host = os.environ.get("RTSP_HOST",     "192.168.1.2")
        pwd  = os.environ.get("RTSP_PASSWORD", "password")
        raw  = raw.replace("RTSP_HOST", host).replace("RTSP_PASSWORD", pwd)
        return json.loads(raw)
    return {"cameras": []}

def save_config(cfg: dict):
    with open(os.path.join(BASE, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

CONFIG = load_config()

# ══════════════════════════════════════════════════════════════════════
#  Telegram
# ══════════════════════════════════════════════════════════════════════

def tg_send(text: str, photo: bytes | None = None,
            bot_token: str = "", chat_id: str = ""):
    tok = bot_token or TELEGRAM_BOT_TOKEN
    cid = chat_id  or TELEGRAM_CHAT_ID
    if not tok or not cid:
        return
    def _do():
        try:
            base = f"https://api.telegram.org/bot{tok}"
            if photo:
                httpx.post(f"{base}/sendPhoto",
                    data={"chat_id": cid, "caption": text, "parse_mode": "HTML"},
                    files={"photo": ("snap.jpg", photo, "image/jpeg")},
                    timeout=10)
            else:
                httpx.post(f"{base}/sendMessage",
                    json={"chat_id": cid, "text": text, "parse_mode": "HTML"},
                    timeout=10)
        except Exception as e:
            log.warning(f"Telegram: {e}")
    threading.Thread(target=_do, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════
#  YOLO singleton
# ══════════════════════════════════════════════════════════════════════

_yolo = None
_yolo_ready = False
_yolo_lock  = threading.Lock()

def get_yolo():
    global _yolo, _yolo_ready
    with _yolo_lock:
        if _yolo is None:
            try:
                import torch
                if hasattr(torch.serialization, "add_safe_globals"):
                    try:
                        from ultralytics.nn.tasks import DetectionModel
                        torch.serialization.add_safe_globals([DetectionModel])
                    except Exception:
                        pass
                import logging as _l
                _l.getLogger("ultralytics").setLevel(_l.WARNING)
                from ultralytics import YOLO
                _yolo = YOLO("yolov8n.pt")
                _yolo(np.zeros((240,320,3), dtype=np.uint8), verbose=False, classes=[0])
                _yolo_ready = True
                log.info("YOLOv8 pronto")
            except Exception as e:
                log.warning(f"YOLO non disponibile: {e}")
    return _yolo if _yolo_ready else None

def yolo_detect(frame: np.ndarray, conf: float) -> list[dict]:
    model = get_yolo()
    if not model:
        return []
    h, w = frame.shape[:2]
    scale = min(640/w, 640/h, 1.0)
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    with _yolo_lock:
        res = model(small, verbose=False, classes=[0], conf=conf)
    out = []
    if res and res[0].boxes is not None:
        for box in res[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            out.append({"bbox":[int(x1/scale),int(y1/scale),
                                 int(x2/scale),int(y2/scale)],
                        "conf": float(box.conf[0])})
    return out

# ══════════════════════════════════════════════════════════════════════
#  Stato camera
# ══════════════════════════════════════════════════════════════════════

class CameraState:
    def __init__(self, cam: dict):
        self.id      = cam["id"]
        self.name    = cam["name"]
        self.url     = cam["url"]
        self.zones   = cam.get("motion_zones", [])
        self.enabled = cam.get("enabled", True)

        self._frame_lock = threading.Lock()
        self.last_frame: bytes | None = None
        self.last_raw:   np.ndarray | None = None

        self._evt_lock = threading.Lock()
        self.events: list[dict] = []

        self._sse_lock   = threading.Lock()
        self._sse_queues: list[asyncio.Queue] = []

        self.connected = False

        # Per escalation: zone attivate di recente {zone_name: timestamp}
        self._zone_hits: dict[str, float] = {}
        self._hits_lock = threading.Lock()

    def push_event(self, level: str, msg: str):
        ev = {"time": datetime.now().strftime("%H:%M:%S"),
              "cam":  self.name, "level": level, "msg": msg}
        with self._evt_lock:
            self.events.append(ev)
            if len(self.events) > 200:
                self.events.pop(0)
        with self._sse_lock:
            for q in self._sse_queues:
                try: q.put_nowait(ev)
                except Exception: pass

    def add_sse(self, q): 
        with self._sse_lock: self._sse_queues.append(q)
    def rm_sse(self, q):
        with self._sse_lock:
            try: self._sse_queues.remove(q)
            except ValueError: pass

    def record_zone_hit(self, zone_name: str):
        with self._hits_lock:
            self._zone_hits[zone_name] = time.time()
            # Rimuovi hit più vecchi di 5 minuti
            cutoff = time.time() - 300
            self._zone_hits = {k:v for k,v in self._zone_hits.items() if v > cutoff}

    def check_escalation(self, escalation_cfg: list) -> str | None:
        """
        escalation_cfg: [{"zones":["Zona A","Zona B"],"window_sec":60,"message":"Alert escalation!"}]
        Ritorna il messaggio se la sequenza è soddisfatta, altrimenti None.
        """
        if not escalation_cfg:
            return None
        now = time.time()
        with self._hits_lock:
            for rule in escalation_cfg:
                req_zones  = rule.get("zones", [])
                window     = rule.get("window_sec", 60)
                msg        = rule.get("message", "🚨 Escalation rilevata!")
                if all(self._zone_hits.get(z, 0) > now - window
                       for z in req_zones):
                    return msg
        return None

CAMERAS: dict[int, CameraState] = {}

# ══════════════════════════════════════════════════════════════════════
#  Worker camera
# ══════════════════════════════════════════════════════════════════════

def motion_analysis_worker(state: CameraState, buf: dict):
    """Thread separato: analizza i frame senza bloccare la live."""
    zone_bg: dict[int, cv2.BackgroundSubtractor] = {}
    zone_bg_keys: dict[int, str] = {}
    zone_cooldown_ts: dict[int, float] = {}
    zone_consec: dict[int, int] = {}
    zone_post: dict[int, dict] = {}

    while buf.get("active", True):
        with buf["lock"]:
            frame = buf.get("frame")
            if frame is not None:
                buf["frame"] = None
        if frame is None:
            time.sleep(0.05)
            continue
        if not state.zones:
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now_t = time.time()

        # Accumula post-evento se in corso (per tutte le zone attive)
        for zi in list(zone_post.keys()):
            post = zone_post[zi]
            post["frames"].append(frame.copy())
            if len(post["frames"]) >= post["target"]:
                all_frames = post["pre_frames"] + post["frames"]
                _send_clip(state, zi, all_frames, post["fps"], post["zone"],
                           post["vid_before"], post["vid_after"])
                del zone_post[zi]
                log.info(f"[Cam {state.id}] Clip zona {zi} completata: "
                         f"{len(all_frames)} frame ({post['vid_before']+post['vid_after']}s)")

        for zi, zone in enumerate(state.zones):
            if not zone.get("enabled", True): continue
            pts = zone.get("points", [])
            if len(pts) < 3: continue

            min_area    = zone.get("min_area",    500)
            cooldown    = zone.get("cooldown",    10)
            use_yolo    = zone.get("detect_persons", False)
            yolo_conf   = zone.get("person_conf", 25) / 100.0
            actions     = zone.get("actions", {})
            zname       = zone.get("name", f"Zona {zi+1}")
            sensitivity = zone.get("sensitivity", 25)
            bg_history  = zone.get("bg_history",  500)
            min_frames  = zone.get("min_frames",  1)
            blur_size   = zone.get("blur_size",   21)
            erode_iter  = zone.get("erode_iter",  0)
            # Parametri video clip
            send_video    = actions.get("send_video", False)
            vid_before    = actions.get("video_before_sec", 10)
            vid_after     = actions.get("video_after_sec",  10)
            analysis_fps  = PROCESS_FPS

            bg_key = f"{zi}_{sensitivity}_{bg_history}"
            if zi not in zone_bg or zone_bg_keys.get(zi) != bg_key:
                zone_bg[zi]      = cv2.createBackgroundSubtractorMOG2(bg_history, sensitivity, False)
                zone_bg_keys[zi] = bg_key
                zone_consec[zi]  = 0

            zmask = np.zeros((h, w), dtype=np.uint8)
            poly  = np.array([[int(p[0]*w), int(p[1]*h)] for p in pts], np.int32)
            cv2.fillPoly(zmask, [poly], 255)

            bsz  = blur_size | 1
            gm   = cv2.GaussianBlur(gray, (bsz, bsz), 0)
            gm   = cv2.bitwise_and(gm, gm, mask=zmask)
            diff = zone_bg[zi].apply(gm)
            if erode_iter > 0:
                diff = cv2.erode(diff, None, iterations=erode_iter)
            diff = cv2.dilate(diff, None, iterations=2)
            diff = cv2.bitwise_and(diff, diff, mask=zmask)

            cnts, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_a   = max((cv2.contourArea(c) for c in cnts), default=0)

            if max_a >= min_area:
                zone_consec[zi] = zone_consec.get(zi, 0) + 1
            else:
                zone_consec[zi] = 0

            if zone_consec.get(zi, 0) < min_frames:
                continue

            person_boxes = []
            if use_yolo:
                boxes = yolo_detect(frame, yolo_conf)
                pn = np.array([[p[0], p[1]] for p in pts], np.float32)
                for b in boxes:
                    cx_n = (b["bbox"][0]+b["bbox"][2])/2/w
                    cy_n = (b["bbox"][1]+b["bbox"][3])/2/h
                    if cv2.pointPolygonTest(pn, (float(cx_n), float(cy_n)), False) >= 0:
                        person_boxes.append(b)
                if not person_boxes:
                    continue

            last = zone_cooldown_ts.get(zi, 0)
            if now_t - last < cooldown:
                continue
            if send_video and zi in zone_post:
                continue
            zone_cooldown_ts[zi] = now_t
            zone_consec[zi]      = 0

            ts    = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            pfx   = "🚶 " if use_yolo else "🔴 "
            state.push_event("motion", f"{pfx}[{zname}] Movimento alle {ts}")

            _, jpgbuf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            snap = jpgbuf.tobytes()

            if actions.get("snapshot"):
                sdir = os.path.join(BASE, "snapshots")
                os.makedirs(sdir, exist_ok=True)
                fname = f"cam{state.id}_{zname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                with open(os.path.join(sdir, fname), "wb") as f:
                    f.write(snap)

            tg_cfg = actions.get("telegram", {})
            if tg_cfg.get("enabled"):
                tok = tg_cfg.get("bot_token") or TELEGRAM_BOT_TOKEN
                cid = tg_cfg.get("chat_id")   or TELEGRAM_CHAT_ID
                msg = f"📷 <b>{state.name}</b>\n🔴 <b>{zname}</b>\n⏰ {ts}"
                # Foto immediata
                if not send_video:
                    tg_send(msg, snap if tg_cfg.get("send_photo", True) else None, tok, cid)
                elif zi not in zone_post:
                    n_pre  = max(1, int(vid_before * analysis_fps))
                    n_post = max(1, int(vid_after  * analysis_fps))
                    with buf["lock"]:
                        ring_frames = [f for _, f in buf["clip_ring"]]
                    if len(ring_frames) >= n_pre:
                        pre_frames = ring_frames[-n_pre:]
                    elif ring_frames:
                        pre_frames = [ring_frames[0]] * (n_pre - len(ring_frames)) + ring_frames
                    else:
                        pre_frames = [frame.copy()] * n_pre
                    zone_post[zi] = {
                        "frames":     [],
                        "pre_frames": pre_frames,
                        "target":     n_post,
                        "fps":        float(analysis_fps),
                        "zone":       zone,
                        "vid_before": vid_before,
                        "vid_after":  vid_after,
                    }
                    log.info(f"[Cam {state.id}] Avvio clip zona {zi}: "
                             f"{n_pre}+{n_post} frame ({vid_before}+{vid_after}s)")

            state.record_zone_hit(zname)
            esc_msg = state.check_escalation(zone.get("escalation", []))
            if esc_msg:
                state.push_event("motion", f"🚨 ESCALATION: {esc_msg}")
                tg_send(f"🚨 <b>ESCALATION</b> – {state.name}\n{esc_msg}\n⏰ {ts}", snap)


def _send_clip(state: CameraState, zi: int, frames: list,
               fps: float, zone: dict,
               vid_before: int = 10, vid_after: int = 10):
    """Assembla i frame in MP4 H.264 via FFmpeg e manda su Telegram."""
    if not frames:
        return
    actions = zone.get("actions", {})
    tg_cfg  = actions.get("telegram", {})
    if not tg_cfg.get("enabled"):
        return

    tok = tg_cfg.get("bot_token") or TELEGRAM_BOT_TOKEN
    cid = tg_cfg.get("chat_id")   or TELEGRAM_CHAT_ID
    if not tok or not cid:
        return

    def _do():
        import subprocess, tempfile
        out_path = None
        try:
            h, w = frames[0].shape[:2]
            if w > 640:
                scale = 640 / w
                w2 = 640
                h2 = int(h * scale)
            else:
                w2, h2 = w, h
            w2 -= w2 % 2
            h2 -= h2 % 2

            fps_i = max(1, min(30, int(round(fps))))
            raw_chunks = []
            for fr in frames:
                fr2 = fr if fr.shape[1] == w2 and fr.shape[0] == h2 else cv2.resize(fr, (w2, h2))
                if not fr2.flags["C_CONTIGUOUS"]:
                    fr2 = np.ascontiguousarray(fr2)
                raw_chunks.append(fr2.tobytes())

            fd, out_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}",
                "-r", str(fps_i), "-i", "pipe:0",
                "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.1",
                "-vsync", "cfr", "-r", str(fps_i),
                "-movflags", "+faststart",
                out_path,
            ]
            result = subprocess.run(
                cmd, input=b"".join(raw_chunks),
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                log.warning(f"[Cam {state.id}] FFmpeg error: {result.stderr.decode()[:300]}")
                return

            if len(frames) > 1:
                diff = float(np.mean(np.abs(
                    frames[0].astype(np.int16) - frames[-1].astype(np.int16))))
                log.info(f"[Cam {state.id}] Clip {len(frames)} frame, diff primo/ultimo: {diff:.1f}")

            zname = zone.get("name", f"Zona {zi+1}")
            ts    = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            total_sec = int(vid_before) + int(vid_after)
            cap   = (f"🎥 <b>{state.name}</b> – <b>{zname}</b>\n"
                     f"⏰ {ts}\n"
                     f"⏱ {total_sec}s ({vid_before}s prima + {vid_after}s dopo)")

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            base = f"https://api.telegram.org/bot{tok}"
            r = httpx.post(f"{base}/sendVideo",
                       data={"chat_id": cid, "caption": cap, "parse_mode": "HTML"},
                       files={"video": ("clip.mp4", video_bytes, "video/mp4")},
                       timeout=120)
            if r.status_code == 200:
                log.info(f"[Cam {state.id}] Video inviato su Telegram ({len(video_bytes)//1024}KB)")
            else:
                log.warning(f"[Cam {state.id}] Telegram sendVideo error: {r.text[:200]}")

        except Exception as e:
            log.warning(f"[Cam {state.id}] Clip error: {e}")
        finally:
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)

    threading.Thread(target=_do, daemon=True).start()


def camera_worker(state: CameraState):
    """Thread principale: legge frame e pubblica SUBITO senza aspettare l'analisi."""
    retry = 5

    while True:
        state.connected = False
        state.push_event("info", "Connessione in corso...")

        url = state.url
        if url.isdigit():
            cap = cv2.VideoCapture(int(url))
        elif os.path.isfile(url):
            cap = cv2.VideoCapture(url)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        ok = False
        for _ in range(16):
            if cap.isOpened():
                r, _ = cap.read()
                if r: ok = True; break
            time.sleep(0.5)

        if not ok:
            cap.release()
            state.push_event("error", "Connessione fallita, riprovo...")
            time.sleep(retry)
            continue

        state.connected = True
        state.push_event("info", "Connesso")
        log.info(f"[Cam {state.id}] {url}")

        is_file  = os.path.isfile(url)
        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = 1.0 / src_fps if is_file else 0
        last_t   = time.time()

        import collections
        process_interval = 1.0 / PROCESS_FPS
        last_process_t   = 0.0
        ring_maxlen      = max(1, int(65 * PROCESS_FPS))
        buf = {
            "frame": None,
            "clip_ring": collections.deque(maxlen=ring_maxlen),
            "lock": threading.Lock(),
            "active": True,
        }
        threading.Thread(target=motion_analysis_worker, args=(state, buf),
                         daemon=True).start()

        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_t = time.time()
                    continue
                break

            # Pubblica il frame SUBITO
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (640, int(h*640/w))) if w > 640 else frame
            _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with state._frame_lock:
                state.last_frame = jpg.tobytes()
                state.last_raw   = frame

            now = time.time()
            if now - last_process_t >= process_interval:
                fc = frame.copy()
                with buf["lock"]:
                    buf["frame"] = fc
                    buf["clip_ring"].append((now, fc.copy()))
                last_process_t = now

            if is_file and interval:
                nt = last_t + interval
                sl = nt - time.time()
                if sl > 0: time.sleep(sl)
                last_t = max(nt, time.time())

        buf["active"] = False
        cap.release()
        state.push_event("error", "Stream perso, riconnessione...")
        time.sleep(retry)

# ══════════════════════════════════════════════════════════════════════
#  Startup
# ══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    for i, cam in enumerate(CONFIG.get("cameras", [])):
        if not cam.get("enabled", True): continue
        state = CameraState(cam)
        CAMERAS[cam["id"]] = state
        def _start(s, d):
            time.sleep(d)
            camera_worker(s)
        threading.Thread(target=_start, args=(state, i*8), daemon=True).start()
    threading.Thread(target=get_yolo, daemon=True).start()
    log.info(f"{len(CAMERAS)} telecamere avviate")

# ══════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    cams = [{"id": s.id, "name": s.name} for s in CAMERAS.values()]
    return templates.TemplateResponse(
        request=request, name="index.html", context={"cameras": cams})


@app.get("/stream/{cam_id}")
async def stream(cam_id: int):
    state = CAMERAS.get(cam_id)
    if not state:
        return JSONResponse({"error": "not found"}, 404)
    async def gen():
        while True:
            with state._frame_lock:
                jpg = state.last_frame
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(1.0 / STREAM_FPS)
    return StreamingResponse(gen(),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/events")
async def sse(request: Request):
    q: asyncio.Queue = asyncio.Queue(100)
    for s in CAMERAS.values(): s.add_sse(q)
    async def gen() -> AsyncGenerator[str, None]:
        hist = []
        for s in CAMERAS.values():
            with s._evt_lock: hist.extend(s.events[-5:])
        for ev in sorted(hist, key=lambda e: e["time"]):
            yield f"data: {json.dumps(ev)}\n\n"
        try:
            while True:
                if await request.is_disconnected(): break
                try:
                    ev = await asyncio.wait_for(q.get(), 15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            for s in CAMERAS.values(): s.rm_sse(q)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/cameras")
async def api_cameras():
    return [{"id": s.id, "name": s.name, "connected": s.connected,
             "zones": len(s.zones)} for s in CAMERAS.values()]


@app.get("/api/cameras/{cam_id}/zones")
async def get_zones(cam_id: int):
    s = CAMERAS.get(cam_id)
    if not s: return JSONResponse({"error":"not found"}, 404)
    return s.zones


@app.post("/api/cameras/{cam_id}/zones")
async def set_zones(cam_id: int, request: Request):
    s = CAMERAS.get(cam_id)
    if not s: return JSONResponse({"error":"not found"}, 404)
    zones = await request.json()
    s.zones = zones
    for cam in CONFIG.get("cameras", []):
        if cam["id"] == cam_id:
            cam["motion_zones"] = zones; break
    save_config(CONFIG)
    log.info(f"[Cam {cam_id}] {len(zones)} zone salvate")
    return {"status": "ok", "zones": len(zones)}


@app.get("/api/snapshot/{cam_id}")
async def snapshot(cam_id: int):
    s = CAMERAS.get(cam_id)
    if not s: return JSONResponse({"error":"not found"}, 404)
    with s._frame_lock:
        jpg = s.last_frame
    if not jpg: return JSONResponse({"error":"no frame"}, 503)
    return StreamingResponse(iter([jpg]), media_type="image/jpeg")


@app.get("/api/events")
async def api_events(limit: int = 100):
    all_ev = []
    for s in CAMERAS.values():
        with s._evt_lock: all_ev.extend(s.events)
    return sorted(all_ev, key=lambda e: e["time"], reverse=True)[:limit]
