"""Authentication for the web UI via WorkOS AuthKit.

The login flow redirects to AuthKit; the callback exchanges the code for the
user, checks the optional email allowlist, and sets a signed session cookie. API
routes depend on `require_user`, which 401s when the cookie is missing/invalid.

WorkOS and itsdangerous are imported lazily so the package imports without the
`web` extra installed.
"""

from __future__ import annotations

import logging
from typing import Optional

from starlette.requests import Request

from ..config import settings

logger = logging.getLogger(__name__)

COOKIE_NAME = "ra_session"
SESSION_MAX_AGE = 7 * 24 * 3600  # 7 days


def _serializer():
    from itsdangerous import URLSafeTimedSerializer

    secret = settings.web_session_secret or "insecure-dev-secret-change-me"
    return URLSafeTimedSerializer(secret, salt="ra-web-session")


def seal_session(data: dict) -> str:
    return _serializer().dumps(data)


def read_session(token: str | None) -> Optional[dict]:
    if not token:
        return None
    from itsdangerous import BadSignature, SignatureExpired

    try:
        return _serializer().loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def _workos_client():
    from workos import WorkOSClient

    return WorkOSClient(
        api_key=settings.workos_api_key, client_id=settings.workos_client_id
    )


def configured() -> bool:
    return bool(settings.workos_api_key and settings.workos_client_id)


def authorization_url(state: str = "") -> str:
    """AuthKit hosted login URL to redirect the browser to."""
    return _workos_client().user_management.get_authorization_url(
        provider="authkit", redirect_uri=settings.web_redirect_uri, state=state or None,
    )


def authenticate_code(code: str) -> dict:
    """Exchange the callback code for the user; raise if not allowed."""
    res = _workos_client().user_management.authenticate_with_code(code=code)
    user = res.user
    email = (getattr(user, "email", "") or "").lower()
    allow = settings.allowed_emails
    if allow and email not in allow:
        raise PermissionError(f"{email} is not on the allowlist")
    name = " ".join(
        p for p in [getattr(user, "first_name", ""), getattr(user, "last_name", "")] if p
    )
    return {"email": email, "name": name or email}


def require_user(request: Request) -> dict:
    """FastAPI dependency: return the session user or 401."""
    from fastapi import HTTPException

    user = read_session(request.cookies.get(COOKIE_NAME))
    if not user:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user
