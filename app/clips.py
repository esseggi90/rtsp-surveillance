"""Assemblaggio clip video e invio Telegram."""

import logging
import os
import subprocess
import tempfile
import threading
from datetime import datetime

import cv2
import httpx
import numpy as np

from app.camera import CameraState
from app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from app.yolo import draw_detections

log = logging.getLogger("surveillance")


def send_clip(state: CameraState, zi: int, frames: list,
              fps: float, zone: dict,
              vid_before: int = 10, vid_after: int = 10,
              frame_dets: list | None = None):
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
        out_path = None
        try:
            h, w = frames[0].shape[:2]
            if w > 640:
                w2, h2 = 640, int(h * 640 / w)
            else:
                w2, h2 = w, h
            w2 -= w2 % 2
            h2 -= h2 % 2

            fps_i = max(1, min(30, int(round(fps))))
            raw_chunks = []
            for i, fr in enumerate(frames):
                fr2 = fr if fr.shape[1] == w2 and fr.shape[0] == h2 else cv2.resize(fr, (w2, h2))
                if frame_dets and i < len(frame_dets) and frame_dets[i]:
                    fr2 = draw_detections(fr2, frame_dets[i])
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
