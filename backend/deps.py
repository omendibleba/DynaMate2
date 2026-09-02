"""
backend.deps
────────────
FastAPI dependency accessors for the pool/enhancer built at startup
(backend.main.lifespan) and stored on app.state.
"""

from fastapi import Request


def get_pool(request: Request):
    return request.app.state.pool


def get_enhancer(request: Request):
    return request.app.state.enhancer
