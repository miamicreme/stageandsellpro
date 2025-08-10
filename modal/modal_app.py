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
    # Editor/type-checker only; not imported at runtime on the runner
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse

# ──────────────────────────────────────────────────────────────────────────────
# Config (env tunables with safe defaults)
# ──────────────────────────────────────────────────────────────────────────────
APP_NAME = "stage-sell-pro-pipeline"
VERSION = os.getenv("VERSION", "2025-08-09")

HF_CACHE_NAME  = os.getenv("HF_CACHE_NAME", "ssp-hf-cache")
HF_CACHE_MOUNT = os.getenv("HF_CACHE_MOUNT", "/root/.cache/huggingface")

API_KEY = os.getenv("API_KEY", "")

SDXL_BASE        = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "diffusers/controlnet-canny-sdxl-1.0")

DEFAULT_GUIDANCE = float(os.getenv("GUIDANCE", "5.0"))
DEFAULT_STEPS    = int(os.getenv("STEPS", "28"))
MAX_EDGE         = int(os.getenv("MAX_EDGE", "1536"))
JPEG_QUALITY     = int(os.getenv("JPEG_QUALITY", "92"))

MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", "20"))
REQUEST_TIMEOUTS = (5, 20)  # (connect, read)

# GPU strings per new Modal guidance: "CPU" (None), "A10G", "A100-40GB", "H100"
GPU_TYPE = (os.getenv("GPU_TYPE", "A10G") or "A10G").upper()
if GPU_TYPE not in {"A10G", "A100-40GB", "H100", "CPU"}:
    GPU_TYPE = "A100-40GB" if GPU_TYPE == "A100" else "A10G"
GPU_ARG = None if GPU_TYPE == "CPU" else GPU_TYPE  # strings, not objects

# Safe ranges
MIN_STEPS, MAX_STEPS = 5, 50
MIN_GUIDE, MAX_GUIDE = 1.0, 12.0
MAX_ROOM_STYLE_LEN   = 200

# ──────────────────────────────────────────────────────────────────────────────
# App + NFS + Image
# ──────────────────────────────────────────────────────────────────────────────
app = modal.App(APP_NAME)

HF_CACHE = modal.NetworkFileSystem.from_name(HF_CACHE_NAME, create_if_missing=True)
NFS_MOUNTS = {HF_CACHE_MOUNT: HF_CACHE}  # mount_path -> NFS

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.44.0",
        "diffusers==0.31.0",
        "accelerate==0.33.0",
        "safetensors>=0.4.3",
        "Pillow==10.4.0",
        "opencv-python-headless==4.10.0.84",
        "numpy==2.0.1",
        "requests>=2.32.3",
        # Needed by @modal.fastapi_endpoint (installed in the image, not on the runner)
        "fastapi==0.111.0",
        "python-multipart==0.0.9",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_MOUNT,
            "HUGGINGFACE_HUB_CACHE": HF_CACHE_MOUNT,
            "TRANSFORMERS_CACHE": HF_CACHE_MOUNT,
            "DIFFUSERS_CACHE": HF_CACHE_MOUNT,
            "PYTHONUNBUFFERED": "1",
        }
    )
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resize_long_edge(pil_img, max_edge: int):
    from PIL import Image
    w, h = pil_img.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return pil_img
    r = float(max_edge) / float(long_edge)
    return pil_img.resize((int(round(w * r)), int(round(h * r))), Image.LANCZOS)

def _to_jpeg_bytes(pil_img, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def _canny(image_pil):
    import numpy as np, cv2
    from PIL import Image
    img = np.array(image_pil.convert("RGB"))
    edges = cv2.Canny(img, 100, 200)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))

def _parse_b64(data: str) -> bytes:
    if data.startswith("data:"):
        m = re.match(r"data:[^;]+;base64,(.*)$", data, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            raise ValueError("invalid_data_url")
        data = m.group(1)
    try:
        return base64.b64decode(data, validate=True)
    except Exception:
        return base64.b64decode(data)

def _fetch_image_from_url(url: str, max_mb: int) -> bytes:
    from urllib.parse import urlparse
    import requests

    u = urlparse(url)
    if u.scheme not in {"http", "https"}:
        raise ValueError("unsupported_url_scheme")

    headers = {"User-Agent": f"StageSellPro/{VERSION}"}
    with requests.get(url, headers=headers, timeout=REQUEST_TIMEOUTS, stream=True) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype and not ctype.startswith("application/octet-stream"):
            raise ValueError("not_an_image_response")

        max_bytes = max_mb * 1024 * 1024
        clen = r.headers.get("Content-Length")
        if clen and int(clen) > max_bytes:
            raise ValueError("image_too_large")

        buf = io.BytesIO()
        read = 0
        for chunk in r.iter_content(chunk_size=1024 * 128):
            if not chunk:
                continue
            read += len(chunk)
            if read > max_bytes:
                raise ValueError("image_too_large")
            buf.write(chunk)
        return buf.getvalue()

def _validate_room_style(s: Optional[str]) -> str:
    s = (s or "modern luxury").strip()
    if len(s) > MAX_ROOM_STYLE_LEN:
        s = s[:MAX_ROOM_STYLE_LEN]
    return s

def _clamp_steps(val: Optional[int]) -> int:
    if val is None:
        return DEFAULT_STEPS
    try:
        v = int(val)
    except Exception:
        v = DEFAULT_STEPS
    return max(MIN_STEPS, min(MAX_STEPS, v))

def _clamp_guidance(val: Optional[float]) -> float:
    if val is None:
        return DEFAULT_GUIDANCE
    try:
        v = float(val)
    except Exception:
        v = DEFAULT_GUIDANCE
    return max(MIN_GUIDE, min(MAX_GUIDE, v))

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Loader (cached per worker)
# ──────────────────────────────────────────────────────────────────────────────
_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline, "cached"

    import torch
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    use_cuda = torch.cuda.is_available() and (GPU_ARG is not None)
    dtype = torch.float16 if use_cuda else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, torch_dtype=dtype, use_safetensors=True, cache_dir=HF_CACHE_MOUNT
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_BASE, controlnet=controlnet, torch_dtype=dtype, use_safetensors=True, cache_dir=HF_CACHE_MOUNT
    )

    try: pipe.enable_vae_slicing()
    except Exception: pass
    try:
        if use_cuda:
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
    except Exception: pass

    device_desc = "cpu"
    if use_cuda:
        pipe.to("cuda")
        import torch as _t
        device_desc = f"cuda:{_t.cuda.current_device()}"

    _pipeline = pipe
    return _pipeline, device_desc

# ──────────────────────────────────────────────────────────────────────────────
# Core worker
# ──────────────────────────────────────────────────────────────────────────────
@app.function(
    name="virtual_stage",
    serialized=True,
    image=image,
    timeout=900,
    gpu=GPU_ARG,
    min_containers=1,              # Modal 1.0 rename (was keep_warm)
    network_file_systems=NFS_MOUNTS,
)
def virtual_stage(
    image_bytes: bytes,
    room_style: str = "modern luxury",
    negative_prompt: str = "lowres, blurry",
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
) -> bytes:
    from PIL import Image, UnidentifiedImageError
    import torch

    g     = _clamp_guidance(guidance_scale)
    steps = _clamp_steps(num_inference_steps)

    pipe, _device_desc = _load_pipeline()

    try:
        inp = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("invalid_image_bytes")

    inp = _resize_long_edge(inp, MAX_EDGE)
    control = _canny(inp)

    use_cuda = torch.cuda.is_available() and (GPU_ARG is not None)
    gen_device = "cuda" if use_cuda else "cpu"

    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=gen_device).manual_seed(int(seed))
        except Exception:
            generator = torch.Generator().manual_seed(int(seed))

    with torch.inference_mode():
        result = pipe(
            image=control,
            prompt=f"{_validate_room_style(room_style)}, photorealistic, high quality, interior design",
            negative_prompt=negative_prompt,
            generator=generator,
            guidance_scale=g,
            num_inference_steps=steps,
        )

    return _to_jpeg_bytes(result.images[0], JPEG_QUALITY)

# ──────────────────────────────────────────────────────────────────────────────
# Web endpoints
# ──────────────────────────────────────────────────────────────────────────────
def _auth(request):  # avoid importing FastAPI types at module import time
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        return {"ok": False, "status": 401, "err": "unauthorized"}
    return {"ok": True}

@app.function(
    name="stage",
    serialized=True,
    image=image,
    timeout=900,
    gpu=GPU_ARG,
    min_containers=1,
    network_file_systems=NFS_MOUNTS,
)
@modal.fastapi_endpoint(method="POST")
async def stage(request):  # type: ignore[no-untyped-def]
    from fastapi import Response
    from fastapi.responses import JSONResponse

    """
    POST /stage
    - application/json: { image_b64 | image_url, room_style?, seed?, guidance_scale?, num_inference_steps? }
    - multipart/form-data: file=<binary>, room_style?, seed?, guidance_scale?, num_inference_steps?
    """
    a = _auth(request)
    if not a["ok"]:
        return JSONResponse({"error": a["err"]}, status_code=a["status"])

    try:
        content_type = (request.headers.get("content-type") or "").lower()
        room_style = "modern luxury"
        seed = None
        guidance_scale = None
        num_inference_steps = None
        raw_bytes: Optional[bytes] = None

        if content_type.startswith("multipart/"):
            form = await request.form()
            uploaded = form.get("file")
            if not uploaded:
                return JSONResponse({"error": "missing_file"}, status_code=400)

            room_style = _validate_room_style(form.get("room_style"))
            if form.get("seed") is not None:
                try: seed = int(form.get("seed"))
                except Exception: pass
            if form.get("guidance_scale") is not None:
                try: guidance_scale = float(form.get("guidance_scale"))
                except Exception: pass
            if form.get("num_inference_steps") is not None:
                try: num_inference_steps = int(form.get("num_inference_steps"))
                except Exception: pass

            raw_bytes = await uploaded.read()

        else:
            payload = await request.json()
            room_style = _validate_room_style(payload.get("room_style"))

            seed = payload.get("seed")
            guidance_scale = payload.get("guidance_scale")
            num_inference_steps = payload.get("num_inference_steps")

            if payload.get("image_url"):
                raw_bytes = _fetch_image_from_url(payload["image_url"], MAX_UPLOAD_MB)
            else:
                image_b64 = payload.get("image_b64")
                if not image_b64:
                    return JSONResponse({"error": "missing_image"}, status_code=400)
                raw_bytes = _parse_b64(image_b64)

        if not raw_bytes:
            return JSONResponse({"error": "empty_image_bytes"}, status_code=400)
        if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            return JSONResponse({"error": "image_too_large"}, status_code=413)

        jpeg_bytes: bytes = virtual_stage.remote(
            raw_bytes,
            room_style=room_style,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        return Response(content=jpeg_bytes, media_type="image/jpeg")

    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": "internal_error", "detail": str(e)}, status_code=500)

@app.function(
    name="health",
    serialized=True,
    image=image,
    min_containers=1,
    network_file_systems=NFS_MOUNTS,
)
@modal.fastapi_endpoint(method="GET")
async def health(_request):  # type: ignore[no-untyped-def]
    from fastapi.responses import JSONResponse

    try:
        return JSONResponse(
            {"ok": True, "app": APP_NAME, "version": VERSION, "gpu": GPU_TYPE, "hf_cache_mount": HF_CACHE_MOUNT},
            status_code=200,
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.function(
    name="warm",
    serialized=True,
    image=image,
    gpu=GPU_ARG,
    min_containers=1,
    network_file_systems=NFS_MOUNTS,
)
@modal.fastapi_endpoint(method="POST")
async def warm(_request):  # type: ignore[no-untyped-def]
    from fastapi.responses import JSONResponse

    try:
        _load_pipeline()
        return JSONResponse({"ok": True, "warmed": True}, status_code=200)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

