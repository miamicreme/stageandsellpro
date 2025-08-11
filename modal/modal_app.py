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
MAX_EDGE = int(os.getenv("MAX_EDGE", "1536"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
REQUEST_TIMEOUTS = (5, 20)  # (connect, read)

# GPU strings per Modal guidance: "CPU" (None), "A10G", "A100-40GB", "H100"
GPU_TYPE = (os.getenv("GPU_TYPE", "A10G") or "A10G").upper()
if GPU_TYPE not in {"A10G", "A100-40GB", "H100", "CPU"}:
    GPU_TYPE = "A100-40GB" if GPU_TYPE == "A100" else "A10G"
GPU_ARG = None if GPU_TYPE == "CPU" else GPU_TYPE  # strings, not objects

# Safe ranges
MIN_STEPS, MAX_STEPS = 5, 50
MIN_GUIDE, MAX_GUIDE = 1.0, 12.0
MAX_ROOM_STYLE_LEN = 200

# ──────────────────────────────────────────────────────────────────────────────
# App + NFS + Image
# ──────────────────────────────────────────────────────────────────────────────
app = modal.App(APP_NAME)

# Persist HF cache between runs
HF_CACHE = modal.NetworkFileSystem.from_name(HF_CACHE_NAME, create_if_missing=True)
NFS_MOUNTS = {HF_CACHE_MOUNT: HF_CACHE}  # mount_path -> NFS

# IMPORTANT: fix the accelerate vs numpy conflict by pinning numpy<2.0
image = (
    modal.Image.debia
