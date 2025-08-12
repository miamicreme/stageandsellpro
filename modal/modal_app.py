# modal/modal_app.py
# Stage & Sell Pro — SDXL + ControlNet virtual staging on Modal (production hardened)

from __future__ import annotations
import base64, io, json, os, time, pathlib
from typing import TYPE_CHECKING, Optional

import modal

if TYPE_CHECKING:
    from fastapi import Request

# ───────────────────────── Config ─────────────────────────
APP_NAME = "stage-sell-pro-pipeline"

SDXL_INPAINT_ID     = os.getenv("SDXL_INPAINT_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
CONTROLNET_MODEL_ID = os.getenv("CONTROLNET_MODEL_ID", "diffusers/controlnet-sdxl-1.0-scribble")
HF_CACHE_NAME       = os.getenv("HF_CACHE_NAME", "ssp-hf-cache")
HF_CACHE_MOUNT      = os.getenv("HF_CACHE_MOUNT", "/root/.cache/huggingface")
GPU_TYPE            = os.getenv("GPU_TYPE", "A10G")  # "A10G" | "A100" | "H100"
API_KEY             = os.getenv("API_KEY", "")
MAX_EDGE            = int(os.getenv("MAX_EDGE", "2048"))
DEFAULT_STEPS       = int(os.getenv("STEPS", "28"))
DEFAULT_GUIDANCE    = float(os.getenv("GUIDANCE", "5.5"))
DEFAULT_SEED        = int(os.getenv("SEED", "0"))

# Keep-warm mode persistence (on NFS so CI calls survive container restarts)
KEEPWARM_DIR  = os.path.join(HF_CACHE_MOUNT, "ssp")
KEEPWARM_FILE = os.path.join(KEEPWARM_DIR, "keepwarm_mode.txt")
DEFAULT_MODE  = "business"  # business | dev | off

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
    """Load SDXL Inpaint + ControlNet + detector once per container."""
    global PIPE, DETECTOR
    if PIPE is not None and DETECTOR is not None:
        return PIPE, DETECTOR

    import torch
    from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
    from controlnet_aux import LineartDetector

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
        torch_dtype=torch.float16
    )

    PIPE = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        SDXL_INPAINT_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    # performance knobs
    PIPE.enable_model_cpu_offload()
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
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

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

# keep-warm state
def _ensure_keepwarm_dir():
    pathlib.Path(KEEPWARM_DIR).mkdir(parents=True, exist_ok=True)

def _get_keepwarm_mode() -> str:
    try:
        with open(KEEPWARM_FILE, "r", encoding="utf-8") as f:
            m = f.read().strip()
            return m if m in ("off", "dev", "business") else DEFAULT_MODE
    except FileNotFoundError:
        return DEFAULT_MODE

def _set_keepwarm_mode(mode: str) -> str:
    mode = (mode or "").lower()
    if mode not in ("off", "dev", "business"):
        mode = DEFAULT_MODE
    _ensure_keepwarm_dir()
    with open(KEEPWARM_FILE, "w", encoding="utf-8") as f:
        f.write(mode)
    return mode

# ───────────────────────── Common function args ─────────────────────────
common_fn_args = dict(
    image=image,
    timeout=900,
    gpu=GPU_TYPE,                                  # ✔ new API: string
    network_file_systems={HF_CACHE_MOUNT: HF_CACHE}# ✔ correct mapping
)

# ───────────────────────── HTTP Endpoints ─────────────────────────
@app.function(**common_fn_args)
@modal.fastapi_endpoint(method="GET", label="health")
async def health() -> dict:
    return {
        "ok": True,
        "app": APP_NAME,
        "models": {"sdxl_inpaint": SDXL_INPAINT_ID, "controlnet": CONTROLNET_MODEL_ID},
        "keepwarm_mode": _get_keepwarm_mode(),
    }

@app.function(**common_fn_args)
@modal.fastapi_endpoint(method="POST", label="stage")
async def stage(request: "Request"):
    err = _check_api_key(request.headers)
    if err:
        return {"error": err}

    form = await request.form()
    file = form.get("image")
    if file is None:
        return {"error": "Missing file field 'image' (multipart/form-data)"}
    payload_raw = form.get("payload") or "{}"
    try:
        payload = json.loads(payload_raw)
    except Exception:
        payload = {"_raw": payload_raw}

    style    = str(payload.get("style", "modern"))
    room     = str(payload.get("room", "living room"))
    steps    = int(payload.get("steps", DEFAULT_STEPS))
    guidance = float(payload.get("guidance", DEFAULT_GUIDANCE))
    seed     = int(payload.get("seed", DEFAULT_SEED)) or None
    strength = float(payload.get("strength", 0.85))
    negative = payload.get("negative_prompt", "blurry, low quality, artifacts, watermark, deformed")
    prompt   = payload.get("prompt") or f"{style} {room}, tasteful furniture, natural light, photo-realistic, 4k, professional interior photograph"

    from PIL import Image, UnidentifiedImageError
    try:
        img_bytes = await file.read()
        base = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return {"error": "Invalid image"}
    base = _resize_long_edge(base, MAX_EDGE)

    t0 = time.time()
    pipe, detector = _lazy_load()
    load_s = time.time() - t0

    t1 = time.time()
    guide = detector(base)                      # grayscale guidance
    mask  = Image.new("L", base.size, 0)        # rely on ControlNet for staging
    prep_s = time.time() - t1

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
            "style": style, "room": room, "steps": steps, "guidance": guidance,
            "seed": seed or 0, "strength": strength, "negative_prompt": negative,
            "prompt": prompt, "sdxl_inpaint": SDXL_INPAINT_ID, "controlnet": CONTROLNET_MODEL_ID,
        },
    }

# Force a warm-up (useful for one-off pings)
@app.function(**common_fn_args)
@modal.fastapi_endpoint(method="GET", label="warm")
async def warm() -> dict:
    _lazy_load()
    return {"ok": True, "warmed": True, "mode": _get_keepwarm_mode()}

# Keep-warm mode setters that your CI expects
@app.function(**common_fn_args)
@modal.fastapi_endpoint(method="POST", label="keepwarm_set")
async def keepwarm_set(request: "Request"):
    # Accept mode via query, form, or JSON
    mode = (request.query_params.get("mode") or "").strip()
    if not mode:
        # try form
        try:
            form = await request.form()
            mode = (form.get("mode") or "").strip()
        except Exception:
            mode = ""
    if not mode:
        try:
            body = await request.body()
            if body:
                data = json.loads(body.decode("utf-8"))
                mode = (data.get("mode") or "").strip()
        except Exception:
            pass
    mode = _set_keepwarm_mode(mode)
    return {"ok": True, "mode": mode}

@app.function(**common_fn_args)
@modal.fastapi_endpoint(method="GET", label="keepwarm_status")
async def keepwarm_status() -> dict:
    return {"ok": True, "mode": _get_keepwarm_mode()}

# Scheduled ping (respects mode)
@app.function(schedule=modal.Period(seconds=300), image=image, network_file_systems={HF_CACHE_MOUNT: HF_CACHE})
def keepwarm_cron():
    mode = _get_keepwarm_mode()
    if mode == "off":
        return {"ok": True, "skipped": True, "mode": mode}
    # Touch the pipeline (lightweight)
    _lazy_load()
    return {"ok": True, "mode": mode}
