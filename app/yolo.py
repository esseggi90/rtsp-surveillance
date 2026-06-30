"""YOLO26 + tracking Ultralytics (ByteTrack) – un modello per camera."""

import logging
import threading

import cv2
import numpy as np

from app.config import YOLO_MODEL

log = logging.getLogger("surveillance")

YOLO_CLASSES: dict[int, str] = {
    0: "persona", 1: "bicicletta", 2: "auto", 3: "moto",
    5: "bus", 7: "camion", 14: "uccello", 15: "gatto", 16: "cane",
}
YOLO_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 220, 0), 1: (255, 200, 0), 2: (0, 165, 255), 3: (255, 100, 0),
    5: (255, 0, 200), 7: (200, 100, 255), 14: (100, 255, 255),
    15: (150, 150, 255), 16: (255, 150, 100),
}

_models: dict[int, tuple[str, object]] = {}
_model_locks: dict[int, threading.Lock] = {}
_registry_lock = threading.Lock()


def default_yolo_cfg() -> dict:
    return {
        "enabled": False,
        "classes": [0, 2],
        "conf": 20,
        "imgsz": 640,
        "fps": 6,
        "model": YOLO_MODEL,
        "show_boxes_live": False,
        "show_boxes_video": True,
    }


def normalize_imgsz(imgsz: int) -> int:
    return 640 if int(imgsz) > 480 else 320


def _lock_for(cam_id: int) -> threading.Lock:
    with _registry_lock:
        if cam_id not in _model_locks:
            _model_locks[cam_id] = threading.Lock()
        return _model_locks[cam_id]


def get_yolo_for_camera(cam_id: int, model_name: str | None = None):
    """Un'istanza YOLO per camera → tracker ByteTrack indipendente."""
    name = model_name or YOLO_MODEL
    with _registry_lock:
        entry = _models.get(cam_id)
        if entry is not None and entry[0] == name:
            return entry[1]

    lock = _lock_for(cam_id)
    with lock:
        with _registry_lock:
            entry = _models.get(cam_id)
            if entry is not None and entry[0] == name:
                return entry[1]
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
            model = YOLO(name)
            model.track(
                np.zeros((480, 640, 3), dtype=np.uint8),
                persist=True, verbose=False, imgsz=640,
            )
            with _registry_lock:
                _models[cam_id] = (name, model)
            log.info(f"YOLO26 pronto cam {cam_id}: {name}")
            return model
        except Exception as e:
            log.warning(f"YOLO non disponibile cam {cam_id} ({name}): {e}")
            return None


def warmup_yolo():
    """Precarica il modello default (startup)."""
    get_yolo_for_camera(0, YOLO_MODEL)


def yolo_track(cam_id: int, frame: np.ndarray, conf: float,
               classes: list[int] | None = None,
               imgsz: int = 640, model_name: str | None = None) -> list[dict]:
    """Tracking live via ultralytics model.track() + ByteTrack."""
    model = get_yolo_for_camera(cam_id, model_name)
    if not model or not classes:
        return []
    imgsz = normalize_imgsz(imgsz)
    half = False
    try:
        import torch
        half = torch.cuda.is_available()
    except Exception:
        pass
    lock = _lock_for(cam_id)
    with lock:
        res = model.track(
            frame, persist=True, verbose=False,
            classes=classes, conf=conf, imgsz=imgsz, iou=0.45,
            half=half, tracker="bytetrack.yaml",
        )
    out = []
    if res and res[0].boxes is not None:
        for box in res[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            tid = int(box.id[0]) if box.id is not None else None
            out.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(box.conf[0]),
                "cls":  cls_id,
                "label": YOLO_CLASSES.get(cls_id, f"cls{cls_id}"),
                "track_id": tid,
            })
    return out


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    if not detections:
        return frame
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cls_id = d.get("cls", 0)
        color = YOLO_CLASS_COLORS.get(cls_id, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tid = d.get("track_id")
        prefix = f"#{tid} " if tid is not None else ""
        tag = f"{prefix}{d.get('label', '?')} {d['conf']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def detections_in_zone(detections: list[dict], pts: list,
                       w: int, h: int, trigger_classes: list[int]) -> list[dict]:
    if not detections or not trigger_classes:
        return []
    pn = np.array([[p[0], p[1]] for p in pts], np.float32)
    matched = []
    for d in detections:
        if d["cls"] not in trigger_classes:
            continue
        x1, y1, x2, y2 = d["bbox"]
        cx_n = (x1 + x2) / 2 / w
        cy_n = (y1 + y2) / 2 / h
        if cv2.pointPolygonTest(pn, (float(cx_n), float(cy_n)), False) >= 0:
            matched.append(d)
    return matched


def zone_trigger_classes(zone: dict) -> list[int] | None:
    tc = zone.get("trigger_classes")
    if tc is not None:
        return tc if tc else None
    if zone.get("detect_persons"):
        return [0]
    return None
