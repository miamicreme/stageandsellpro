# modal_app.py
# Stage & Sell Pro — SDXL + ControlNet virtual staging pipeline on Modal
# (ASGI web endpoint; defer FastAPI imports so the runner doesn't need FastAPI)

from __future__ import annotations
import base64
import io
import os
import json
from typing import Optional, TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import Request as FastAPIRequest

# --- API Key loading from multiple env var names ---
API_KEY = (
    os.environ.get("X_API_KEY")
    or os.environ.get("API_KEY")
    or os.environ.get("STAGE_AND_SELL_API_KEY")
)

# --- Modal secrets attachment ---
image = modal.Image.debian_slim().pip_install(
    "torch", "diffusers", "transformers", "fastapi", "uvicorn"
)

app = modal.App("stage-and-sell-pro")
secret = modal.Secret.from_name("stage-and-sell-secrets")

PIPE = None
DETECTOR = None

def _lazy_load():
    """Load SDXL Inpaint + ControlNet + detector once per container."""
    global PIPE, DETECTOR
    if PIPE is not None and DETECTOR is not None:
        return PIPE, DETECTOR
    # Example: load your pipeline here
    PIPE = "pipeline_loaded"
    DETECTOR = "detector_loaded"
    return PIPE, DETECTOR

@app.function(image=image, secrets=[secret])
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request

    web_app = FastAPI()

    @web_app.middleware("http")
    async def check_api_key(request: FastAPIRequest, call_next):
        provided_key = request.headers.get("x-api-key") or request.query_params.get("api_key")
        if API_KEY and provided_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    @web_app.post("/process")
    async def process_image(request: FastAPIRequest):
        data = await request.json()
        image_b64 = data.get("image")
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image provided")
        # Process with PIPE & DETECTOR
        PIPE, DETECTOR = _lazy_load()
        return {"result": "processed"}

    return web_app
