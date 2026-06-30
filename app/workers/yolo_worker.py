"""Thread YOLO26 + ByteTrack – mai blocca la live."""

import logging
import time

from app.analysis_buffer import AnalysisBuffer
from app.camera import CameraState
from app.config import YOLO_MODEL
from app.yolo import yolo_track

log = logging.getLogger("surveillance")


def yolo_worker(state: CameraState, buf: AnalysisBuffer):
    last_yolo_t = 0.0
    while buf.active:
        yolo_cfg = state.yolo
        if not yolo_cfg.get("enabled"):
            time.sleep(0.1)
            continue

        now = time.time()
        yolo_fps = max(1, min(15, int(yolo_cfg.get("fps", 6))))
        if now - last_yolo_t < 1.0 / yolo_fps:
            time.sleep(0.005)
            continue

        if buf.peek_yolo_frame() is None:
            time.sleep(0.01)
            continue

        frame = buf.take_yolo_frame()
        if frame is None:
            continue
        last_yolo_t = now

        conf    = yolo_cfg.get("conf", 20) / 100.0
        classes = yolo_cfg.get("classes") or [0, 2]
        imgsz   = int(yolo_cfg.get("imgsz", 640))
        model   = yolo_cfg.get("model") or YOLO_MODEL
        dets    = yolo_track(state.id, frame, conf, classes, imgsz, model)
        buf.set_detections(dets)
