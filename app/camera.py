"""Stato runtime di ogni telecamera."""

import asyncio
import threading
import time
from datetime import datetime

from app.config import CONFIG, save_config
from app.yolo import default_yolo_cfg


class CameraState:
    def __init__(self, cam: dict):
        self.id      = cam["id"]
        self.name    = cam["name"]
        self.url     = cam["url"]
        self.zones   = cam.get("motion_zones", [])
        self.enabled = cam.get("enabled", True)
        self.yolo    = {**default_yolo_cfg(), **cam.get("yolo", {})}

        self._frame_lock = threading.Lock()
        self.last_frame: bytes | None = None
        self.last_raw:   object | None = None

        self._evt_lock = threading.Lock()
        self.events: list[dict] = []

        self._sse_lock   = threading.Lock()
        self._sse_queues: list[asyncio.Queue] = []

        self.connected = False
        self._zone_hits: dict[str, float] = {}
        self._hits_lock = threading.Lock()

    def update_yolo(self, cfg: dict):
        self.yolo = {**default_yolo_cfg(), **cfg}
        for cam in CONFIG.get("cameras", []):
            if cam["id"] == self.id:
                cam["yolo"] = self.yolo
                break
        save_config(CONFIG)

    def push_event(self, level: str, msg: str):
        ev = {"time": datetime.now().strftime("%H:%M:%S"),
              "cam":  self.name, "level": level, "msg": msg}
        with self._evt_lock:
            self.events.append(ev)
            if len(self.events) > 200:
                self.events.pop(0)
        with self._sse_lock:
            for q in self._sse_queues:
                try:
                    q.put_nowait(ev)
                except Exception:
                    pass

    def add_sse(self, q):
        with self._sse_lock:
            self._sse_queues.append(q)

    def rm_sse(self, q):
        with self._sse_lock:
            try:
                self._sse_queues.remove(q)
            except ValueError:
                pass

    def record_zone_hit(self, zone_name: str):
        with self._hits_lock:
            self._zone_hits[zone_name] = time.time()
            cutoff = time.time() - 300
            self._zone_hits = {k: v for k, v in self._zone_hits.items() if v > cutoff}

    def check_escalation(self, escalation_cfg: list) -> str | None:
        if not escalation_cfg:
            return None
        now = time.time()
        with self._hits_lock:
            for rule in escalation_cfg:
                req_zones = rule.get("zones", [])
                window    = rule.get("window_sec", 60)
                msg       = rule.get("message", "🚨 Escalation rilevata!")
                if all(self._zone_hits.get(z, 0) > now - window for z in req_zones):
                    return msg
        return None


CAMERAS: dict[int, CameraState] = {}
