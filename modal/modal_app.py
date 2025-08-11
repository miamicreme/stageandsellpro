# modal_app.py
# Stage & Sell Pro — SDXL + ControlNet virtual staging pipeline on Modal
# (ASGI web endpoint; defer FastAPI imports so the runner doesn't need FastAPI)

from __future__ import annotations
import base64
import io
import os
import json
import re
from typing import Optional, TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse

# ──────────────────────────────────────────────────────────────────────────────
# Config (env tunables with safe defaults)
# ──────────────────────────────────────────────────────────────────────────────
APP_NAME = "stage-sell-pro-pipeline"
VERSION = os.getenv("VERSION", "2025-08-09")

HF_CACHE_NAME = os.getenv("HF_CACHE_NAME", "ssp-hf-cache")
HF_CACHE_MOUNT = os.getenv("HF_CACHE_MOUNT", "/root/.cache/huggingface")

API_KEY = os.getenv("API_KEY", "")

SDXL_BASE = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "diffusers/controlnet-canny-sdxl-1.0")

DEFAULT_GUIDANCE = float(os.getenv("GUIDANCE", "5.0"))
DEFAULT_STEPS = int(os.getenv("STEPS", "28"))
MAX_ED_
