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

import logging
import os
import threading
import time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.camera import CAMERAS, CameraState
from app.config import BASE, CONFIG
from app.routes import router
from app.workers.stream import camera_worker
from app.yolo import warmup_yolo

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("surveillance")

app = FastAPI(title="RTSP Surveillance")

static_dir = os.path.join(BASE, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(router)


@app.on_event("startup")
async def startup():
    for i, cam in enumerate(CONFIG.get("cameras", [])):
        if not cam.get("enabled", True):
            continue
        state = CameraState(cam)
        CAMERAS[cam["id"]] = state

        def _start(s, d):
            time.sleep(d)
            camera_worker(s)

        threading.Thread(target=_start, args=(state, i * 8), daemon=True).start()
    threading.Thread(target=warmup_yolo, daemon=True).start()
    log.info(f"{len(CAMERAS)} telecamere avviate")
