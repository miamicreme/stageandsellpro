# modal_app.py
# Stage & Sell Pro — SDXL + ControlNet virtual staging pipeline on Modal
# (ASGI web endpoint, no FastAPI dependency)

from __future__ import annotations
import base64
import io
import os
import json
from typing import Optional, Tuple

import modal

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
APP_NAME = "stage-sell-pro-pipeline"

HF_CACHE_NAME = os.getenv("HF_CACHE_NAME", "ssp-hf-cache")
HF_CACHE_MOUNT = os.getenv("HF_CACHE_MOUNT", "/root/.cache/huggingface")

API_KEY = os.getenv("API_KEY", "")

SDXL_BASE = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "diffusers/controlnet-canny-sdxl-1.0")

DEFAULT_GUIDANCE = float(os.getenv("GUIDANCE", "5.0"))
DEFAULT_STEPS = int(os.getenv("STEPS", "28"))
MAX_EDGE = int(os.getenv("MAX_EDGE", "1536"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))

GPU_TYPE = os.getenv("GPU_TYPE", "A10G").upper()
if GPU_TYPE not in {"A10G", "A100", "H100", "CPU"}:
    GPU_TYPE = "A10G"

# Map to Modal GPU objects (or None for CPU)
from modal import gpu as modal_gpu
_GPU_MAP = {
    "A10G": modal_gpu.A10G(),
    "A100": modal_gpu.A100(),
    "H100": modal_gpu.H100(),
}
GPU_ARG = _GPU_MAP.get(GPU_TYPE)  # None if CPU

# ──────────────────────────────────────────────────────────────────────────────
# App + NFS + Image
# ──────────────────────────────────────────────────────────────────────────────
app = modal.App(APP_NAME)

# Correct NFS API & mapping (NFS -> mount path)
HF_CACHE = modal.NetworkFileSystem.from_name(HF_CACHE_NAME, create_if_missing=True)

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
    # replicate to 3 channels so ControlNet sees an RGB-like image
    return Image.fromarray(np.stack([edges] * 3, axis=-1))

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Loader (cached)
# ──────────────────────────────────────────────────────────────────────────────
_pipeline = None  # cached in worker

def _load_pipeline():
    """
    Returns: (pipe, device_desc)
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline, "cached"

    import torch
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    use_cuda = torch.cuda.is_available() and (GPU_ARG is not None)
    dtype = torch.float16 if use_cuda else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=HF_CACHE_MOUNT,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_BASE,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=HF_CACHE_MOUNT,
    )

    # Memory tweaks
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        if use_cuda:
            pipe.enable_model_cpu_offload()  # better balance for large models
        else:
            pipe.enable_sequential_cpu_offload()
    except Exception:
        pass

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
    name="virtual_stage",  # callable by other functions
    image=image,
    timeout=900,
    gpu=GPU_ARG,
    network_file_systems={HF_CACHE: HF_CACHE_MOUNT},
)
def virtual_stage(
    image_bytes: bytes,
    room_style: str = "modern luxury",
    negative_prompt: str = "lowres, blurry",
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
) -> bytes:
    from PIL import Image
    import torch

    g = float(guidance_scale or DEFAULT_GUIDANCE)
    steps = int(num_inference_steps or DEFAULT_STEPS)

    pipe, device_desc = _load_pipeline()

    inp = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inp = _resize_long_edge(inp, MAX_EDGE)
    control = _canny(inp)

    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
        except Exception:
            generator = torch.Generator().manual_seed(int(seed))

    # Run
    with torch.inference_mode():
        result = pipe(
            image=control,
            prompt=f"{room_style}, photorealistic, high quality, interior design",
            negative_prompt=negative_prompt,
            generator=generator,
            guidance_scale=g,
            num_inference_steps=steps,
        )

    return _to_jpeg_bytes(result.images[0], JPEG_QUALITY)

# ──────────────────────────────────────────────────────────────────────────────
# Web endpoint
# NOTE: Path is derived from function name="stage".
# ──────────────────────────────────────────────────────────────────────────────
@app.function(
    name="stage",  # ⇦ yields POST /stage
    image=image,
    timeout=900,
    gpu=GPU_ARG,
    network_file_systems={HF_CACHE: HF_CACHE_MOUNT},
)
@modal.web_endpoint(method="POST")
async def stage(request: modal.web.Request):
    """
    POST /stage

    Content-Types supported:
    - application/json
        { "image_b64": "...", "room_style": "modern luxury", "seed": 123, "guidance_scale": 5.0, "num_inference_steps": 28 }
        or
        { "image_url": "https://..." , ... }

    - multipart/form-data
        fields: file=<binary>, room_style=?, seed=?, guidance_scale=?, num_inference_steps=?
    """
    try:
        # Simple API key gate (optional)
        if API_KEY and request.headers.get("x-api-key") != API_KEY:
            return modal.web.Response.json({"error": "unauthorized"}, status=401)

        content_type = (request.headers.get("content-type") or "").lower()

        room_style = "modern luxury"
        seed = None
        guidance_scale = None
        num_inference_steps = None
        raw_bytes: Optional[bytes] = None

        if content_type.startswith("multipart/"):
            # Modal Request supports .form() in recent versions; if not available,
            # fall back to raw body parsing on your side.
            form = await request.form()
            uploaded = form.get("file")
            if not uploaded:
                return modal.web.Response.json({"error": "missing_file"}, status=400)

            room_style = form.get("room_style") or room_style
            if form.get("seed") is not None:
                try:
                    seed = int(form.get("seed"))
                except Exception:
                    pass
            if form.get("guidance_scale") is not None:
                try:
                    guidance_scale = float(form.get("guidance_scale"))
                except Exception:
                    pass
            if form.get("num_inference_steps") is not None:
                try:
                    num_inference_steps = int(form.get("num_inference_steps"))
                except Exception:
                    pass

            raw_bytes = await uploaded.read()

        else:
            # Expect JSON
            payload = await request.json()
            room_style = payload.get("room_style", room_style)
            seed = payload.get("seed")
            guidance_scale = payload.get("guidance_scale")
            num_inference_steps = payload.get("num_inference_steps")

            if payload.get("image_url"):
                import requests

                r = requests.get(payload["image_url"], timeout=15)
                r.raise_for_status()
                raw_bytes = r.content
            else:
                image_b64 = payload.get("image_b64")
                if not image_b64:
                    return modal.web.Response.json({"error": "missing_image"}, status=400)
                raw_bytes = base64.b64decode(image_b64)

        # Call the GPU worker
        jpeg_bytes: bytes = virtual_stage.remote(
            raw_bytes,
            room_style=room_style,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        # Return as an image
        return modal.web.Response(jpeg_bytes, content_type="image/jpeg")

    except Exception as e:
        # Structured error
        return modal.web.Response.json({"error": str(e)}, status=500)

# ──────────────────────────────────────────────────────────────────────────────
# Local entry
# ──────────────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("Serve locally for dev:\n  modal serve modal/modal_app.py")
    print("Deployed endpoint path will be POST /stage")
