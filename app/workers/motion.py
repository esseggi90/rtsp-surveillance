"""Thread motion detection – zone, clip e alert Telegram."""

import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np

from app.analysis_buffer import AnalysisBuffer
from app.camera import CameraState
from app.clips import send_clip
from app.config import BASE, PROCESS_FPS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from app.telegram import tg_send
from app.yolo import detections_in_zone, zone_trigger_classes

log = logging.getLogger("surveillance")


def motion_analysis_worker(state: CameraState, buf: AnalysisBuffer):
    """Thread separato: analizza i frame senza bloccare la live."""
    zone_bg: dict[int, cv2.BackgroundSubtractor] = {}
    zone_bg_keys: dict[int, str] = {}
    zone_cooldown_ts: dict[int, float] = {}
    zone_consec: dict[int, int] = {}
    zone_post: dict[int, dict] = {}

    while buf.active:
        frame = buf.take_motion_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        if not state.zones:
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now_t = time.time()

        yolo_cfg = state.yolo
        cam_dets = buf.get_detections()
        buf.append_clip(now_t, frame.copy(), cam_dets)

        for zi in list(zone_post.keys()):
            post = zone_post[zi]
            post["frames"].append(frame.copy())
            post["frame_dets"].append(list(cam_dets))
            if len(post["frames"]) >= post["target"]:
                all_frames = post["pre_frames"] + post["frames"]
                all_dets   = post["pre_dets"] + post["frame_dets"]
                show_boxes = yolo_cfg.get("show_boxes_video", True)
                send_clip(state, zi, all_frames, post["fps"], post["zone"],
                          post["vid_before"], post["vid_after"],
                          all_dets if show_boxes else None)
                del zone_post[zi]
                log.info(f"[Cam {state.id}] Clip zona {zi} completata: "
                         f"{len(all_frames)} frame ({post['vid_before']+post['vid_after']}s)")

        for zi, zone in enumerate(state.zones):
            if not zone.get("enabled", True):
                continue
            pts = zone.get("points", [])
            if len(pts) < 3:
                continue

            min_area    = zone.get("min_area",    500)
            cooldown    = zone.get("cooldown",    10)
            actions     = zone.get("actions", {})
            zname       = zone.get("name", f"Zona {zi+1}")
            sensitivity = zone.get("sensitivity", 25)
            bg_history  = zone.get("bg_history",  500)
            min_frames  = zone.get("min_frames",  1)
            blur_size   = zone.get("blur_size",   21)
            erode_iter  = zone.get("erode_iter",  0)
            trigger_cls = zone_trigger_classes(zone)
            yolo_only   = zone.get("yolo_only", False)
            send_video    = actions.get("send_video", False)
            vid_before    = actions.get("video_before_sec", 10)
            vid_after     = actions.get("video_after_sec",  10)
            analysis_fps  = PROCESS_FPS

            matched_dets: list[dict] = []
            if trigger_cls and yolo_cfg.get("enabled"):
                matched_dets = detections_in_zone(cam_dets, pts, w, h, trigger_cls)

            if yolo_only and trigger_cls and yolo_cfg.get("enabled"):
                if not matched_dets:
                    zone_consec[zi] = 0
                    continue
                zone_consec[zi] = zone_consec.get(zi, 0) + 1
                if zone_consec[zi] < min_frames:
                    continue
                yolo_hit = True
                goto_trigger = True
            else:
                goto_trigger = False
                yolo_hit = False

            if not goto_trigger:
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

                if trigger_cls and yolo_cfg.get("enabled"):
                    if not matched_dets:
                        continue
                    yolo_hit = True

            last = zone_cooldown_ts.get(zi, 0)
            if now_t - last < cooldown:
                continue
            if send_video and zi in zone_post:
                continue
            zone_cooldown_ts[zi] = now_t
            zone_consec[zi]      = 0

            ts = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            if yolo_hit and matched_dets:
                labels = ", ".join(sorted({d["label"] for d in matched_dets}))
                pfx   = f"🎯 {labels}"
            elif yolo_only:
                pfx   = "🎯"
            else:
                pfx   = "🔴"
            state.push_event("motion", f"{pfx} [{zname}] Rilevamento alle {ts}")

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
                if not send_video:
                    tg_send(msg, snap if tg_cfg.get("send_photo", True) else None, tok, cid)
                elif zi not in zone_post:
                    n_pre  = max(1, int(vid_before * analysis_fps))
                    n_post = max(1, int(vid_after  * analysis_fps))
                    ring = buf.snapshot_ring()
                    ring_frames = [f for _, f, _ in ring]
                    ring_dets   = [d for _, _, d in ring]
                    if len(ring_frames) >= n_pre:
                        pre_frames = ring_frames[-n_pre:]
                        pre_dets   = ring_dets[-n_pre:]
                    elif ring_frames:
                        pad = n_pre - len(ring_frames)
                        pre_frames = [ring_frames[0]] * pad + ring_frames
                        pre_dets   = [[],] * pad + ring_dets
                    else:
                        pre_frames = [frame.copy()] * n_pre
                        pre_dets   = [list(cam_dets)] * n_pre
                    zone_post[zi] = {
                        "frames":     [],
                        "frame_dets": [],
                        "pre_frames": pre_frames,
                        "pre_dets":   pre_dets,
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
