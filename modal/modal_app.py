# modal_app.py
# Stage & Sell Pro — SDXL + ControlNet virtual staging pipeline on Modal (ASGI version without FastAPI dependency)

from __future__ import annotations
import base64
import io
import os
import json
from typing import Optional, Tuple

import modal

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
GPU_ARG: Optional[str] = None if GPU_TYPE == "CPU" else GPU_TYPE

app = modal.App(APP_NAME)

# Create or load the shared HF cache NFS
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

def _resize_long_edge(pil_img, max_edge: int):
    from PIL import Image
    w, h = pil_img.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return pil_img
    ratio = max_edge / float(long_edge)
    return pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

def _to_jpeg_bytes(pil_img, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def _canny(image_pil):
    import numpy as np, cv2
    from PIL import Image
    img = np.array(image_pil.convert("RGB"))
    edges = cv2.Canny(img, 100, 200)
    # replicate single channel to 3 channels as expected by ControlNet
    return Image.fromarray(np.stack([edges] * 3, axis=-1))

_pipeline = None

def _load_pipeline():
    """Lazy-load and cache the SDXL + ControlNet pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline, ("cuda" if GPU_ARG else "cpu")

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

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        # Prefer CPU offload model mgmt when not on CUDA; otherwise use CUDA with CPU offload for memory balance
        pipe.enable_sequential_cpu_offload() if not use_cuda else pipe.enable_model_cpu_offload()
    except Exception:
        pass

    if use_cuda:
        pipe.to("cuda")

    _pipeline = pipe
    return _pipeline, ("cuda" if use_cuda else "cpu")

@app.function(
    image=image,
    timeout=900,
    # IMPORTANT: dict is {NetworkFileSystem: mount_path}
    network_file_systems={HF_CACHE: HF_CACHE_MOUNT},
    gpu=GPU_ARG,
)
@modal.web_endpoint(method="POST", label="stage")
async def stage_http(request):
    try:
        # Simple header key check
        if API_KEY and request.headers.get("x-api-key") != API_KEY:
            return modal.Response(
                json.dumps({"error": "unauthorized"}).encode(),
                content_type="application/json",
                status_code=401,
            )

        content_type = (request.headers.get("content-type") or "").lower()
        room_style = "modern luxury"
        raw_bytes = None

        if "multipart/form-data" in content_type:
            form = await request.form()
            uploaded = form.get("file")
            if not uploaded:
                return modal.Response(
                    json.dumps({"error": "missing_file"}).encode(),
                    content_type="application/json",
                    status_code=400,
                )
            room_style = form.get("room_style") or room_style
            raw_bytes = await uploaded.read()
        else:
            payload = await request.json()
            room_style = payload.get("room_style", room_style)
            if payload.get("image_url"):
                import requests

                r = requests.get(payload["image_url"], timeout=10)
                r.raise_for_status()
                raw_bytes = r.content
            else:
                image_b64 = payload.get("image_b64")
                if not image_b64:
                    return modal.Response(
                        json.dumps({"error": "missing_image"}).encode(),
                        content_type="application/json",
                        status_code=400,
                    )
                raw_bytes = base64.b64decode(image_b64)

        out = virtual_stage.remote(raw_bytes, room_style)

        # If the worker returned a JSON error blob for any reason, forward with 500
        try:
            maybe = json.loads(out.decode("utf-8"))
            if isinstance(maybe, dict) and maybe.get("error"):
                return modal.Response(out, content_type="application/json", status_code=500)
        except Exception:
            pass

        return modal.Response(out, content_type="image/jpeg")

    except Exception as e:
        return modal.Response(
            json.dumps({"error": str(e)}).encode(),
            content_type="application/json",
            status_code=500,
        )

@app.function(
    image=image,
    timeout=900,
    network_file_systems={HF_CACHE: HF_CACHE_MOUNT},
    gpu=GPU_ARG,
)
def virtual_stage(
    image: bytes,
    room_style: str = "modern luxury",
    negative_prompt: str = "lowres, blurry",
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
) -> bytes:
    from PIL import Image
    import torch

    g = guidance_scale or DEFAULT_GUIDANCE
    steps = num_inference_steps or DEFAULT_STEPS

    pipe, device = _load_pipeline()

    inp = Image.open(io.BytesIO(image)).convert("RGB")
    inp = _resize_long_edge(inp, MAX_EDGE)
    control = _canny(inp)

    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        except Exception:
            generator = torch.Generator().manual_seed(int(seed))

    with torch.inference_mode():
        result = pipe(
            image=control,
            prompt=f"{room_style}, photorealistic, high detail, interior design",
            negative_prompt=negative_prompt,
            generator=generator,
            guidance_scale=g,
            num_inference_steps=steps,
        )

    return _to_jpeg_bytes(result.images[0], JPEG_QUALITY)

@app.local_entrypoint()
def main():
    print("Use: modal run modal_app.py::stage_http  (for local test HTTP endpoint)")
