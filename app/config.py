"""Configurazione e variabili d'ambiente."""

import json
import os

BASE = os.path.dirname(os.path.dirname(__file__))

STREAM_FPS         = int(os.environ.get("STREAM_FPS",    "10"))
PROCESS_FPS        = int(os.environ.get("PROCESS_FPS",   "6"))
YOLO_MODEL         = os.environ.get("YOLO_MODEL", "yolo26s.pt")
JPEG_QUALITY       = int(os.environ.get("JPEG_QUALITY",  "60"))
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")


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
