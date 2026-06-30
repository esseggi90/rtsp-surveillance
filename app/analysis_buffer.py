"""Buffer condiviso tra stream e pipeline di analisi."""

import collections
import threading

from app.config import PROCESS_FPS


class AnalysisBuffer:
    def __init__(self, ring_seconds: int = 65):
        maxlen = max(1, int(ring_seconds * PROCESS_FPS))
        self._lock = threading.Lock()
        self.active = True
        self.yolo_frame = None
        self.motion_frame = None
        self.last_dets: list[dict] = []
        self.clip_ring: collections.deque = collections.deque(maxlen=maxlen)

    def stop(self):
        self.active = False

    def push_yolo(self, frame):
        with self._lock:
            self.yolo_frame = frame

    def push_motion(self, frame):
        with self._lock:
            self.motion_frame = frame

    def peek_yolo_frame(self):
        with self._lock:
            return self.yolo_frame

    def take_motion_frame(self):
        with self._lock:
            frame = self.motion_frame
            self.motion_frame = None
        return frame

    def take_yolo_frame(self):
        with self._lock:
            frame = self.yolo_frame
            self.yolo_frame = None
        return frame

    def set_detections(self, dets: list[dict]):
        with self._lock:
            self.last_dets = dets

    def get_detections(self) -> list[dict]:
        with self._lock:
            return list(self.last_dets)

    def append_clip(self, ts: float, frame, dets: list[dict]):
        with self._lock:
            self.clip_ring.append((ts, frame, dets))

    def snapshot_ring(self) -> list:
        with self._lock:
            return list(self.clip_ring)
