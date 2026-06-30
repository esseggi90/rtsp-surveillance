"""Thread stream – live fluida; YOLO/track in background."""

import logging
import os
import threading
import time

import cv2

from app.analysis_buffer import AnalysisBuffer
from app.camera import CameraState
from app.config import JPEG_QUALITY, PROCESS_FPS, STREAM_FPS
from app.workers.motion import motion_analysis_worker
from app.workers.yolo_worker import yolo_worker
from app.yolo import draw_detections

log = logging.getLogger("surveillance")


def _open_capture(url: str) -> cv2.VideoCapture:
    if url.isdigit():
        return cv2.VideoCapture(int(url))
    if os.path.isfile(url):
        return cv2.VideoCapture(url)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap


def stream_worker(state: CameraState, buf: AnalysisBuffer):
    retry = 5
    encode_iv = 1.0 / STREAM_FPS
    motion_iv = 1.0 / PROCESS_FPS
    last_encode = 0.0
    last_motion = 0.0

    while buf.active:
        cap = _open_capture(state.url)
        ok = False
        for _ in range(16):
            if not buf.active:
                cap.release()
                return
            if cap.isOpened():
                r, _ = cap.read()
                if r:
                    ok = True
                    break
            time.sleep(0.5)

        if not ok:
            cap.release()
            state.connected = False
            time.sleep(retry)
            continue

        state.connected = True
        log.info(f"[Cam {state.id}] {state.url}")

        is_file = os.path.isfile(state.url)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        file_iv = 1.0 / src_fps if is_file else 0
        last_file_t = time.time()

        while buf.active:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_file_t = time.time()
                    continue
                break

            now = time.time()
            yolo_cfg = state.yolo
            yolo_on = yolo_cfg.get("enabled", False)

            if yolo_on and now - last_motion >= motion_iv:
                buf.push_yolo(frame.copy())
                buf.push_motion(frame.copy())
                last_motion = now

            if now - last_encode >= encode_iv:
                display = frame
                if yolo_on and yolo_cfg.get("show_boxes_live", False):
                    dets = buf.get_detections()
                    if dets:
                        display = draw_detections(frame, dets)
                h, w = display.shape[:2]
                small = cv2.resize(display, (640, int(h * 640 / w))) if w > 640 else display
                _, jpg = cv2.imencode(".jpg", small,
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                with state._frame_lock:
                    state.last_frame = jpg.tobytes()
                    state.last_raw = frame
                last_encode = now

            if is_file and file_iv:
                nt = last_file_t + file_iv
                sl = nt - time.time()
                if sl > 0:
                    time.sleep(sl)
                last_file_t = max(nt, time.time())

        cap.release()
        state.connected = False
        time.sleep(retry)


def camera_worker(state: CameraState):
    retry = 5
    while True:
        state.connected = False
        state.push_event("info", "Connessione in corso...")

        buf = AnalysisBuffer(ring_seconds=65)
        threads = [
            threading.Thread(target=stream_worker, args=(state, buf), daemon=True),
            threading.Thread(target=yolo_worker, args=(state, buf), daemon=True),
            threading.Thread(target=motion_analysis_worker, args=(state, buf), daemon=True),
        ]
        for t in threads:
            t.start()

        for _ in range(30):
            if state.connected:
                state.push_event("info", "Connesso")
                break
            time.sleep(0.5)
        else:
            buf.stop()
            for t in threads:
                t.join(timeout=2)
            state.push_event("error", "Connessione fallita, riprovo...")
            time.sleep(retry)
            continue

        while buf.active and state.connected:
            time.sleep(2)

        buf.stop()
        state.push_event("error", "Stream perso, riconnessione...")
        for t in threads:
            t.join(timeout=3)
        time.sleep(retry)
