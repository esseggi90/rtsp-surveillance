"""Invio messaggi Telegram."""

import logging
import threading

import httpx

from app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger("surveillance")


def tg_send(text: str, photo: bytes | None = None,
            bot_token: str = "", chat_id: str = ""):
    tok = bot_token or TELEGRAM_BOT_TOKEN
    cid = chat_id  or TELEGRAM_CHAT_ID
    if not tok or not cid:
        return

    def _do():
        try:
            base = f"https://api.telegram.org/bot{tok}"
            if photo:
                httpx.post(f"{base}/sendPhoto",
                    data={"chat_id": cid, "caption": text, "parse_mode": "HTML"},
                    files={"photo": ("snap.jpg", photo, "image/jpeg")},
                    timeout=10)
            else:
                httpx.post(f"{base}/sendMessage",
                    json={"chat_id": cid, "text": text, "parse_mode": "HTML"},
                    timeout=10)
        except Exception as e:
            log.warning(f"Telegram: {e}")

    threading.Thread(target=_do, daemon=True).start()
