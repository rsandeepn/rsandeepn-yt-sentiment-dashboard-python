import hashlib
import html
import logging
import os
import secrets
from urllib.parse import quote

import requests


logger = logging.getLogger(__name__)
RESEND_EMAILS_URL = "https://api.resend.com/emails"


def new_reset_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def reset_url(token: str) -> str:
    frontend_url = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173").rstrip("/")
    return f"{frontend_url}/reset-password?token={quote(token, safe='')}"


def password_reset_expiry_minutes() -> int:
    return max(5, min(int(os.getenv("PASSWORD_RESET_EXPIRE_MINUTES", "30")), 120))


def password_reset_cooldown_seconds() -> int:
    return max(0, min(int(os.getenv("PASSWORD_RESET_COOLDOWN_SECONDS", "60")), 3600))


def send_password_reset_email(recipient: str, url: str) -> None:
    if os.getenv("PASSWORD_RESET_LOG_LINK", "false").strip().lower() == "true":
        logger.warning("Local password reset link for %s: %s", recipient, url)
        return

    api_key = os.getenv("RESEND_API_KEY", "").strip()
    sender = os.getenv("PASSWORD_RESET_FROM_EMAIL", "").strip()
    if not api_key or not sender:
        raise RuntimeError(
            "Password reset email is not configured. Set RESEND_API_KEY and "
            "PASSWORD_RESET_FROM_EMAIL."
        )

    expiry_minutes = password_reset_expiry_minutes()
    safe_url = html.escape(url, quote=True)
    response = requests.post(
        RESEND_EMAILS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": sender,
            "to": [recipient],
            "subject": "Reset your CommentScope password",
            "html": (
                "<p>We received a request to reset your CommentScope password.</p>"
                f'<p><a href="{safe_url}">Reset your password</a></p>'
                f"<p>This link expires in {expiry_minutes} minutes and can only be used once.</p>"
                "<p>If you did not request this change, you can ignore this email.</p>"
            ),
            "text": (
                "We received a request to reset your CommentScope password.\n\n"
                f"Reset your password: {url}\n\n"
                f"This link expires in {expiry_minutes} minutes and can only be used once.\n"
                "If you did not request this change, you can ignore this email."
            ),
        },
        timeout=10,
    )
    response.raise_for_status()


def deliver_password_reset_email_safely(recipient: str, url: str) -> None:
    try:
        send_password_reset_email(recipient, url)
    except Exception:
        logger.exception("Unable to deliver password reset email to %s", recipient)
