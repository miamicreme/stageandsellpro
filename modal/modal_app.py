# modal/modal_app.py
# Stage & Sell Pro — SDXL + ControlNet virtual staging on Modal (production hardened)

from __future__ import annotations
import base64, io, json, os, time
from typing import TYPE_CHECKING, Optional

import modal

if TYPE_CHECKING:
    from fastapi import Request

# ───────────────────────── Config ─────────────────────────
APP_NAME = "stage-sell-pro-pipeline"

# Models
SDXL_INPAINT_ID     = os.getenv("SDXL_INPAINT_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
CONTROLNET_MODEL_ID = os.getenv("CONTROLNET_MODEL_ID", "diffusers/controlnet-sdxl-1.0-scribble")  # lineart/scribble-like
HF_CACHE_NAME       = os.getenv("HF_CACHE_NAME", "ssp-hf-cache")
HF_CACHE_MOUNT      = os.getenv("HF_CACHE_MOUNT", "/root/.cache/huggingface")
GPU_TYPE            = os.getenv("GPU_TYPE", "A10G")  # use strings: "A10G", "A100", "H100"
API_KEY             = os.getenv("API_KEY", "")
MAX_EDGE            = int(os.getenv("MAX_EDGE", "2048"))
DEFAULT_STEPS       = int(os.getenv("STEPS", "28"))
DEFAULT_GUIDANCE    = float(os.getenv("GUIDANCE", "5.5"))
DEFAULT_SEED        = int(os.getenv("SEED", "0"))

# ───────────────────────── Modal setup ─────────────────────────
app = modal.App(APP_NAME)

HF_CACHE = modal.NetworkFileSystem.from_name(HF_CACHE_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.2.2",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "safetensors>=0.4.2",
        "diffusers>=0.27.2",
        "controlnet-aux>=0.0.9",
        "Pillow>=10.2.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.29.0",
        "starlette>=0.37.2",
        "numpy>=1.26.4",
        "tqdm>=4.66.2",
        "moviepy>=1.0.3",
        "supabase>=2.4.3",
    )
    .env({"HF_HOME": HF_CACHE_MOUNT})
)

# ───────────────────────── Model holder ─────────────────────────
PIPE = None
DETECTOR = None

def _lazy_load():
    """Load SDXL Inpaint + ControlNet and the lineart detector once per container."""
    global PIPE, DETECTOR
    if PIPE is not None and DETECTOR is not None:
        return PIPE, DETECTOR

    import torch
    from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
    from controlnet_aux import LineartDetector

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
