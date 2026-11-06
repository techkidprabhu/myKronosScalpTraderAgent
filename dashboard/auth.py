"""
Token authentication for the dashboard (Jupyter-style).

The token comes from DASHBOARD_TOKEN in .env, or is generated once and
kept in data/.dashboard_token (mode 0600). Opening
http://host:port/?token=<token> swaps it for an HttpOnly, SameSite=Strict
cookie, so the token isn't left in the address bar and no other website
can make the browser send it (blocks CSRF and DNS-rebinding attacks).
Scripts can instead send `Authorization: Bearer <token>`.
"""
import hmac
import os
import secrets
from pathlib import Path

from flask import Flask, Response, redirect, request

COOKIE_NAME = "dashboard_token"
COOKIE_MAX_AGE = 30 * 24 * 3600
PUBLIC_PREFIXES = ("/static/",)


def load_or_create_token(token_file: Path) -> str:
    env_token = os.environ.get("DASHBOARD_TOKEN", "").strip()
    if env_token:
        return env_token
    if token_file.exists():
        return token_file.read_text().strip()
    token = secrets.token_urlsafe(32)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    return token


def _presented_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer "):].strip()
    return request.cookies.get(COOKIE_NAME, "")


def _same_origin() -> bool:
    """Reject state-changing requests whose Origin header names a different
    site. Defence in depth on top of the SameSite cookie."""
    origin = request.headers.get("Origin")
    if origin is None:
        return True  # non-browser client (curl, tests)
    return origin.split("://", 1)[-1] == request.host


def install(app: Flask, token: str):
    @app.before_request
    def require_token():
        if request.path.startswith(PUBLIC_PREFIXES):
            return None

        query_token = request.args.get("token")
        if query_token is not None and request.method == "GET":
            if hmac.compare_digest(query_token, token):
                resp = redirect(request.path)
                resp.set_cookie(
                    COOKIE_NAME, token, max_age=COOKIE_MAX_AGE,
                    httponly=True, samesite="Strict", secure=request.is_secure,
                )
                return resp
            return _unauthorized()

        if not hmac.compare_digest(_presented_token(), token):
            return _unauthorized()

        if request.method not in ("GET", "HEAD", "OPTIONS") and not _same_origin():
            return Response("Cross-origin request blocked", status=403)
        return None


def _unauthorized():
    if request.path.startswith("/api/"):
        return Response('{"error": "unauthorized"}', status=401, mimetype="application/json")
    return Response(
        "<h3>Dashboard login required</h3>"
        "<p>Open the login link printed in the terminal when the dashboard "
        "started (<code>http://&lt;host&gt;:&lt;port&gt;/?token=...</code>).</p>",
        status=401, mimetype="text/html",
    )
