"""API configuration and middleware."""
from __future__ import annotations

import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

MAX_BODY_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "65536"))


def get_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key gate when GRAY_MATTER_API_KEY is set."""

    def __init__(self, app, api_key: str | None):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if not self.api_key:
            return await call_next(request)

        if request.url.path in ("/health", "/", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        provided = request.headers.get("X-API-Key") or request.headers.get(
            "Authorization", ""
        ).removeprefix("Bearer ").strip()
        if provided != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
            )
        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            if len(body) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large."},
                )
            request._body = body
        return await call_next(request)
