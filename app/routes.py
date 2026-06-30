"""Route FastAPI – dashboard, stream, API."""

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.camera import CAMERAS
from app.config import BASE, CONFIG, STREAM_FPS, save_config

log = logging.getLogger("surveillance")
templates = Jinja2Templates(directory=f"{BASE}/templates")
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    cams = [{"id": s.id, "name": s.name} for s in CAMERAS.values()]
    return templates.TemplateResponse(
        request=request, name="index.html", context={"cameras": cams})


@router.get("/stream/{cam_id}")
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


@router.get("/events")
async def sse(request: Request):
    q: asyncio.Queue = asyncio.Queue(100)
    for s in CAMERAS.values():
        s.add_sse(q)

    async def gen() -> AsyncGenerator[str, None]:
        hist = []
        for s in CAMERAS.values():
            with s._evt_lock:
                hist.extend(s.events[-5:])
        for ev in sorted(hist, key=lambda e: e["time"]):
            yield f"data: {json.dumps(ev)}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), 15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            for s in CAMERAS.values():
                s.rm_sse(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/api/cameras")
async def api_cameras():
    return [{"id": s.id, "name": s.name, "connected": s.connected,
             "zones": len(s.zones), "yolo": s.yolo.get("enabled", False)}
            for s in CAMERAS.values()]


@router.get("/api/cameras/{cam_id}/yolo")
async def get_yolo_settings(cam_id: int):
    s = CAMERAS.get(cam_id)
    if not s:
        return JSONResponse({"error": "not found"}, 404)
    return s.yolo


@router.post("/api/cameras/{cam_id}/yolo")
async def set_yolo_settings(cam_id: int, request: Request):
    s = CAMERAS.get(cam_id)
    if not s:
        return JSONResponse({"error": "not found"}, 404)
    cfg = await request.json()
    s.update_yolo(cfg)
    log.info(f"[Cam {cam_id}] YOLO aggiornato: enabled={s.yolo.get('enabled')}")
    return {"status": "ok", "yolo": s.yolo}


@router.get("/api/cameras/{cam_id}/zones")
async def get_zones(cam_id: int):
    s = CAMERAS.get(cam_id)
    if not s:
        return JSONResponse({"error": "not found"}, 404)
    return s.zones


@router.post("/api/cameras/{cam_id}/zones")
async def set_zones(cam_id: int, request: Request):
    s = CAMERAS.get(cam_id)
    if not s:
        return JSONResponse({"error": "not found"}, 404)
    zones = await request.json()
    s.zones = zones
    for cam in CONFIG.get("cameras", []):
        if cam["id"] == cam_id:
            cam["motion_zones"] = zones
            break
    save_config(CONFIG)
    log.info(f"[Cam {cam_id}] {len(zones)} zone salvate")
    return {"status": "ok", "zones": len(zones)}


@router.get("/api/snapshot/{cam_id}")
async def snapshot(cam_id: int):
    s = CAMERAS.get(cam_id)
    if not s:
        return JSONResponse({"error": "not found"}, 404)
    with s._frame_lock:
        jpg = s.last_frame
    if not jpg:
        return JSONResponse({"error": "no frame"}, 503)
    return StreamingResponse(iter([jpg]), media_type="image/jpeg")


@router.get("/api/events")
async def api_events(limit: int = 100):
    all_ev = []
    for s in CAMERAS.values():
        with s._evt_lock:
            all_ev.extend(s.events)
    return sorted(all_ev, key=lambda e: e["time"], reverse=True)[:limit]
