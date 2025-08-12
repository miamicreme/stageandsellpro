# modal/modal_app.py
# Stage & Sell Pro — SDXL + ControlNet virtual staging on Modal (production hardened)
# - FastAPI HTTP endpoints via @modal.fastapi_endpoint
# - SDXL Inpaint + ControlNet-SDXL (scribble/lineart-like) guidance
# - Hugging Face cache on Modal NFS, GPU, keep-warm, health
# - Multipart upload: file field "image", JSON payload field "payload"
# - Returns: JSON with output_base64, timings, params

from __future__ import annotations
import base64, io, json, os, time
from typing import TYPE_CHECKING, Optional

import modal

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import JSONResponse

# ───────────────────────── Config ─────────────────────────
APP_NAME = "stage-sell-pro-pipeline"

# Models (change safely via env)
SDXL_INPAINT_ID     = os.getenv("SDXL_INPAINT_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
CONTROLNET_MODEL_ID = os.getenv("CONTROLNET_MODEL_ID", "diffusers/controlnet-sdxl-1.0-scribble")  # “lineart-like”
HF_CACHE_NAME       = os.getenv("HF_CACHE_NAME", "ssp-hf-cache")
HF_CACHE_MOUNT      = os.getenv("HF_CACHE_MOUNT", "/root/.cache/huggingface")
GPU_TYPE            = os.getenv("GPU_TYPE", "A10G")  # A10G/A100/H100
API_KEY             = os.getenv("API_KEY", "")       # optional x-api-key check
MAX_EDGE            = int(os.getenv("MAX_EDGE", "2048"))
DEFAULT_STEPS       = int(os.getenv("STEPS", "28"))
DEFAULT_GUIDANCE    = float(os.getenv("GUIDANCE", "5.5"))
DEFAULT_SEED        = int(os.getenv("SEED", "0"))    # 0 => random

# ───────────────────────── Modal setup ─────────────────────────
app = modal.App(APP_NAME)

HF_CACHE = modal.NetworkFileSystem.from_name(HF_CACHE_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")  # for potential video assembly, optional
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
        "moviepy>=1.0.3",   # optional
        "supabase>=2.4.3",  # optional; integrate if you want storage uploads
    )
    .env({"HF_HOME": HF_CACHE_MOUNT})
)

# ───────────────────────── Model holder ─────────────────────────
# We keep a singleton in the container to avoid reloading models between calls.
PIPE = None
DETECTOR = None

def _lazy_load():
    global PIPE, DETECTOR
    if PIPE is not None and DETECTOR is not None:
        return PIPE, DETECTOR

    import torch
    from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
    from controlnet_aux import LineartDetector  # scribble/lineart-like preprocessor
    from PIL import Image

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
        torch_dtype=torch.float16
    )

    PIPE = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        SDXL_INPAINT_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    # memory/performance knobs
    PIPE.enable_model_cpu_offload()  # good default on A10G
    PIPE.enable_vae_slicing()
    try:
        PIPE.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    DETECTOR = LineartDetector.from_pretrained("lllyasviel/Annotators")
    return PIPE, DETECTOR

# ───────────────────────── Helpers ─────────────────────────
def _resize_long_edge(img, max_edge=2048):
    from PIL import Image
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.LANCZOS)

def _b64_jpeg(img, quality=92) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _check_api_key(hdrs) -> Optional[str]:
    if not API_KEY:
        return None
    if hdrs.get("x-api-key") != API_KEY:
        return "Unauthorized: bad or missing x-api-key"
    return None

# ───────────────────────── HTTP Endpoints ─────────────────────────
@app.function(image=image, timeout=900, gpu=modal.gpu.A10G() if GPU_TYPE=="A10G" else modal.gpu.A100(), network_file_systems={HF_CACHE: HF_CACHE_MOUNT})
@modal.fastapi_endpoint(method="GET", label="health")
async def health() -> dict:
    return {"ok": True, "app": APP_NAME, "models": {"sdxl_inpaint": SDXL_INPAINT_ID, "controlnet": CONTROLNET_MODEL_ID}}

@app.function(image=image, timeout=900, gpu=modal.gpu.A10G() if GPU_TYPE=="A10G" else modal.gpu.A100(), network_file_systems={HF_CACHE: HF_CACHE_MOUNT})
@modal.fastapi_endpoint(method="POST", label="stage")
async def stage(request: "Request"):
    # API key (optional)
    err = _check_api_key(request.headers)
    if err:
        return {"error": err}

    # Parse multipart form
    form = await request.form()
    file = form.get("image")
    if file is None:
        return {"error": "Missing file field 'image' (multipart/form-data)"}
    payload_raw = form.get("payload") or "{}"
    try:
        payload = json.loads(payload_raw)
    except Exception:
        payload = {"_raw": payload_raw}

    style     = str(payload.get("style", "modern"))
    room      = str(payload.get("room", "living room"))
    steps     = int(payload.get("steps", DEFAULT_STEPS))
    guidance  = float(payload.get("guidance", DEFAULT_GUIDANCE))
    seed      = int(payload.get("seed", DEFAULT_SEED)) or None
    strength  = float(payload.get("strength", 0.85))  # how much to respect mask
    negative  = payload.get("negative_prompt", "blurry, low quality, artifacts, watermark, deformed")
    prompt    = payload.get("prompt") or f"{style} {room}, tasteful furniture, natural light, photo-realistic, 4k, professional interior photograph"

    # Load image
    from PIL import Image, UnidentifiedImageError
    try:
        img_bytes = await file.read()
        base = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return {"error": "Invalid image"}
    base = _resize_long_edge(base, MAX_EDGE)

    # Models
    t0 = time.time()
    pipe, detector = _lazy_load()
    load_s = time.time() - t0

    # Preprocess: lineart/scribble-like guidance
    t1 = time.time()
    guide = detector(base)  # grayscale guidance map
    # For inpainting, we also need a mask. If none provided, detect empty areas via lineart gaps (simple heuristic).
    # Here we use a full canvas (no hole) and rely on ControlNet guidance to “stage” the room.
    mask = Image.new("L", base.size, 0)  # 0=keep, 255=paint; if you want object removal, compute a true mask here.
    prep_s = time.time() - t1

    # Inference
    import torch
    gen = torch.Generator("cuda") if seed is not None else None
    if seed is not None:
        gen.manual_seed(seed)

    t2 = time.time()
    with torch.autocast("cuda"):
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=base,
            mask_image=mask,
            control_image=guide,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen,
            strength=strength,
        ).images[0]
    infer_s = time.time() - t2

    # Return base64 and timings
    return {
        "ok": True,
        "output_base64": _b64_jpeg(out, quality=92),
        "timings": {
            "load_models_s": round(load_s, 3),
            "preprocess_s": round(prep_s, 3),
            "inference_s": round(infer_s, 3),
            "total_s": round(time.time() - t0, 3),
        },
        "params": {
            "style": style,
            "room": room,
            "steps": steps,
            "guidance": guidance,
            "seed": seed or 0,
            "strength": strength,
            "negative_prompt": negative,
            "prompt": prompt,
            "sdxl_inpaint": SDXL_INPAINT_ID,
            "controlnet": CONTROLNET_MODEL_ID,
        }
    }

# Keep-warm: ping the health route periodically to keep container hot.
@app.function(schedule=modal.Period(seconds=300), image=image, network_file_systems={HF_CACHE: HF_CACHE_MOUNT})
def keepwarm_cron():
    # no-op; presence keeps image warm & the container alive
    return {"ok": True}
