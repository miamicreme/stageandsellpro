# modal/modal_app.py
# Stage & Sell Pro — SDXL + ControlNet virtual staging on Modal (production hardened, credit-friendly)

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
import pathlib
from typing import Optional, Tuple

import modal

# ── Config ────────────────────────────────────────────────────────────────────
APP_NAME = "stage-sell-pro-pipeline"

# Models
SDXL_INPAINT_ID = os.getenv("SDXL_INPAINT_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
CONTROLNET_MODEL_ID = os.getenv("CONTROLNET_MODEL_ID", "diffusers/controlnet-sdxl-1.0-scribble")

# Hugging Face cache on Modal NFS
HF_CACHE_NAME = os.getenv("HF_CACHE_NAME", "ssp-hf-cache")
HF_CACHE_MOUNT = os.getenv("HF_CACHE_MOUNT", "/root/.cache/huggingface")

# GPU type per new Modal API (string; e.g., "A10G", "A100", "H100")
GPU_TYPE = (os.getenv("GPU_TYPE") or "A10G").upper()
if GPU_TYPE not in {"A10G", "A100", "H100"}:
    GPU_TYPE = "A10G"

# Auth
API_KEY = os.getenv("API_KEY", "").strip()

# Inference defaults
MAX_EDGE = int(os.getenv("MAX_EDGE", "2048"))
DEFAULT_STEPS = int(os.getenv("STEPS", "28"))
DEFAULT_GUIDANCE = float(os.getenv("GUIDANCE", "5.5"))
DEFAULT_SEED = int(os.getenv("SEED", "0"))

# CORS (optional): set CORS_ORIGIN="*" or to your site origin
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "").strip()

# Keep-warm mode persisted on NFS (survives container restarts)
# NOTE: Default to OFF to conserve credits. Flip via /keepwarm-set.
KEEPWARM_DIR = os.path.join(HF_CACHE_MOUNT, "ssp")
KEEPWARM_FILE = os.path.join(KEEPWARM_DIR, "keepwarm_mode.txt")
DEFAULT_MODE = (os.getenv("KEEPWARM_DEFAULT_MODE") or "off").lower()  # off | dev | business

# Reduce scheduler frequency to save credits (env override allowed)
KEEPWARM_PERIOD_SEC = int(os.getenv("KEEPWARM_PERIOD_SEC", "1800"))  # default: 30 min

# Optional Supabase upload
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY", "")).strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "ssp-outputs").strip()

# ── Modal setup ───────────────────────────────────────────────────────────────
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

# Shared function args (split to avoid reserving GPU for admin endpoints)
gpu_fn_args = dict(
    image=image,
    timeout=900,
    gpu=GPU_TYPE,                                   # ✔ new Modal API — pass string
    network_file_systems={HF_CACHE_MOUNT: HF_CACHE} # ✔ correct NFS mapping
)
cpu_fn_args = dict(
    image=image,
    timeout=900,
    network_file_systems={HF_CACHE_MOUNT: HF_CACHE}
)

# ── Globals (lazy singletons inside container) ───────────────────────────────
PIPE = None
DETECTOR = None
_SB = None  # Supabase client

# ── Utilities ─────────────────────────────────────────────────────────────────
def _json(data: dict, status: int = 200):
    from starlette.responses import JSONResponse
    headers = {}
    if CORS_ORIGIN:
        headers.update({
            "Access-Control-Allow-Origin": CORS_ORIGIN,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, x-api-key",
        })
    return JSONResponse(content=data, status_code=status, headers=headers)

def _ok(payload: dict, status: int = 200):
    payload = {"ok": True, **payload}
    return _json(payload, status=status)

def _err(msg: str, code: str = "bad_request", status: int = 400):
    return _json({"ok": False, "error": msg, "error_code": code}, status=status)

def _check_api_key(hdrs) -> Optional[str]:
    if not API_KEY:
        return None
    if hdrs.get("x-api-key") != API_KEY:
        return "Unauthorized: bad or missing x-api-key"
    return None

def _ensure_keepwarm_dir():
    pathlib.Path(KEEPWARM_DIR).mkdir(parents=True, exist_ok=True)

def _get_keepwarm_mode() -> str:
    try:
        with open(KEEPWARM_FILE, "r", encoding="utf-8") as f:
            m = f.read().strip().lower()
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

def _resize_long_edge(img, max_edge=2048):
    from PIL import Image
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def _b64_jpeg(img, quality=92) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _sb_client():
    global _SB
    if _SB is not None:
        return _SB
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None
    from supabase import create_client
    _SB = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _SB

def _maybe_upload_jpeg(jpeg_bytes: bytes) -> Optional[str]:
    sb = _sb_client()
    if not sb:
        return None
    day = time.strftime("%Y-%m-%d")
    key = f"sdxl/{day}/{uuid.uuid4().hex}.jpg"
    try:
        sb.storage.from_(SUPABASE_BUCKET).upload(key, jpeg_bytes, {"content-type": "image/jpeg", "upsert": "true"})
        return sb.storage.from_(SUPABASE_BUCKET).get_public_url(key)
    except Exception:
        return None

def _parse_payload_dict(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {"_raw": raw}

def _norm_params(p: dict) -> Tuple[str, str, int, float, Optional[int], float, str, str]:
    style = str(p.get("style", "modern"))
    room = str(p.get("room", "living room"))
    steps = max(5, min(75, int(p.get("steps", DEFAULT_STEPS))))
    guidance = max(1.0, min(13.0, float(p.get("guidance", DEFAULT_GUIDANCE))))
    seed = int(p.get("seed", DEFAULT_SEED)) or None
    strength = max(0.0, min(1.0, float(p.get("strength", 0.85))))
    negative = p.get("negative_prompt", "blurry, low quality, artifacts, watermark, deformed")
    prompt = p.get("prompt") or f"{style} {room}, tasteful furniture, natural light, photo-realistic, 4k, professional interior photograph"
    return style, room, steps, guidance, seed, strength, negative, prompt

# ── Model loader ─────────────────────────────────────────────────────────────
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
        torch_dtype=torch.float16,
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

# ── HTTP Endpoints ───────────────────────────────────────────────────────────
# CPU-only admin endpoints (no GPU reserved)
@app.function(**cpu_fn_args)
@modal.fastapi_endpoint(method="GET", label="health")
async def health():
    return _ok({
        "app": APP_NAME,
        "models": {"sdxl_inpaint": SDXL_INPAINT_ID, "controlnet": CONTROLNET_MODEL_ID},
        "gpu": GPU_TYPE,
        "keepwarm_mode": _get_keepwarm_mode(),
        "cors_origin": CORS_ORIGIN or None,
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
    })

@app.function(**cpu_fn_args)
@modal.fastapi_endpoint(method="GET", label="warm")
async def warm():
    try:
        _lazy_load()
    except Exception as e:
        return _err(f"Model load failed: {e}", code="model_load_failed", status=503)
    return _ok({"warmed": True, "mode": _get_keepwarm_mode()})

@app.function(**cpu_fn_args)
@modal.fastapi_endpoint(method="POST", label="keepwarm-set")
async def keepwarm_set(request):
    mode = (request.query_params.get("mode") or "").strip()
    if not mode:
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
    return _ok({"mode": mode})

# Extra alias without dash to make CI typos harmless (URL: /keepwarmset)
@app.function(**cpu_fn_args)
@modal.fastapi_endpoint(method="POST", label="keepwarmset")
async def keepwarmset(request):
    return await keepwarm_set(request)

@app.function(**cpu_fn_args)
@modal.fastapi_endpoint(method="GET", label="keepwarm-status")
async def keepwarm_status():
    return _ok({"mode": _get_keepwarm_mode()})

# GPU-backed inference endpoints
@app.function(**gpu_fn_args)
@modal.fastapi_endpoint(method="POST", label="stage")
async def stage(request):
    return await _stage_impl(request)

# Pretty label alias (use this as "/" on your custom domain)
@app.function(**gpu_fn_args)
@modal.fastapi_endpoint(method="POST", label="stagesellpro")
async def stagesellpro(request):
    return await _stage_impl(request)

# Shared implementation
async def _stage_impl(request):
    # API key
    err = _check_api_key(request.headers)
    if err:
        return _err(err, code="unauthorized", status=401)

    ct = (request.headers.get("content-type") or "").lower()

    # 1) JSON mode
    if "application/json" in ct:
        try:
            body = await request.json()
        except Exception:
            return _err("Invalid JSON body", code="invalid_json", status=400)
        b64 = body.get("image_base64")
        if not (isinstance(b64, str) and b64.strip()):
            return _err("Missing 'image_base64' in JSON", code="missing_image", status=400)
        payload = body.get("payload") if isinstance(body.get("payload"), dict) else _parse_payload_dict(body.get("payload", "{}"))
        try:
            img_bytes = base64.b64decode(b64.split(",")[-1], validate=False)
        except Exception:
            return _err("image_base64 is not valid base64", code="invalid_base64", status=400)

    # 2) Multipart mode
    else:
        try:
            form = await request.form()
        except Exception:
            return _err("Expected multipart/form-data or application/json", code="unsupported_media_type", status=415)
        file = form.get("image")
        if file is None:
            return _err("Missing file field 'image' (multipart/form-data)", code="missing_image", status=400)
        payload = _parse_payload_dict(form.get("payload") or "{}")
        try:
            img_bytes = await file.read()
        except Exception:
            return _err("Could not read uploaded file", code="read_error", status=400)

    # Decode image
    from PIL import Image, UnidentifiedImageError
    try:
        base = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return _err("Invalid image format", code="invalid_image", status=400)
    base = _resize_long_edge(base, MAX_EDGE)

    # Normalize params
    style, room, steps, guidance, seed, strength, negative, prompt = _norm_params(payload)

    # Load models
    t0 = time.time()
    try:
        pipe, detector = _lazy_load()
    except Exception as e:
        return _err(f"Model load failed: {e}", code="model_load_failed", status=503)
    load_s = time.time() - t0

    # Preprocess
    t1 = time.time()
    try:
        guide = detector(base)
    except Exception as e:
        return _err(f"Preprocess failed: {e}", code="preprocess_failed", status=500)

    from PIL import Image as _PILImage
    mask = _PILImage.new("L", base.size, 0)  # rely on ControlNet for “staging”
    prep_s = time.time() - t1

    # Inference
    import torch
    gen = torch.Generator(device="cuda") if seed is not None else None
    if seed is not None:
        gen.manual_seed(seed)

    t2 = time.time()
    try:
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
    except Exception as e:
        return _err(f"Inference failed: {e}", code="inference_failed", status=500)
    infer_s = time.time() - t2

    # Encode / upload
    jpeg_buf = io.BytesIO()
    out.save(jpeg_buf, format="JPEG", quality=92, optimize=True)
    jpeg_bytes = jpeg_buf.getvalue()
    b64_out = base64.b64encode(jpeg_bytes).decode("ascii")
    out_url = _maybe_upload_jpeg(jpeg_bytes)

    return _ok({
        "output_base64": b64_out,
        "output_url": out_url,
        "images": [b64_out],
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
    })

# Scheduled keep-warm — slower & safe by default (CPU)
@app.function(schedule=modal.Period(seconds=KEEPWARM_PERIOD_SEC), **cpu_fn_args)
def keepwarm_cron():
    mode = _get_keepwarm_mode()
    # No-op unless explicitly enabled to conserve credits
    if mode == "off":
        return {"ok": True, "skipped": True, "mode": mode}
    # Light-touch ping in dev
    if mode == "dev":
        return {"ok": True, "ping": True, "mode": mode}
    # business: touch the pipeline to keep weights warm on the next GPU invoke
    try:
        _lazy_load()
        return {"ok": True, "mode": mode, "warmed": True}
    except Exception as e:
        # Prevent error spam in dashboard
        return {"ok": False, "mode": mode, "error": f"{e.__class__.__name__}: {e}"}
