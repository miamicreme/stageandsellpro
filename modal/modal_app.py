import os, io, shutil, tempfile, time, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

import modal
from supabase import create_client, Client
from PIL import Image, UnidentifiedImageError

# ─────────────────────────── Config / Tunables ───────────────────────────────
APP_NAME = "stage-sell-pro-pipeline"
MAX_EDGE = int(os.getenv("MAX_EDGE", "2048"))          # limit long edge (px)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))
POLL_PERIOD_SECONDS = int(os.getenv("POLL_PERIOD_SECONDS", "120"))
GPU_TYPE = os.getenv("GPU_TYPE", "A10G")               # A10G, A100, H100, or "cpu"
HF_CACHE_NAME = os.getenv("HF_CACHE_NAME", "ssp-hf-cache")
HF_CACHE_MOUNT = os.getenv("HF_CACHE_MOUNT", "/root/.cache/huggingface")
ASSETS_BUCKET = os.getenv("ASSETS_BUCKET", "assets")
UPLOADS_BUCKET = os.getenv("UPLOADS_BUCKET", "uploads")

# ─────────────────────────── Modal App & Image ───────────────────────────────
app = modal.App(APP_NAME)

# Newer SDK: use from_name(create_if_missing=True)
HF_CACHE = modal.NetworkFileSystem.from_name(HF_CACHE_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        # GPU + Diffusers (pinned)
        "torch==2.2.2", "torchvision==0.17.2", "torchaudio==2.2.2", "xformers==0.0.25",
        "diffusers==0.28.0", "transformers==4.41.0", "accelerate==0.28.0", "controlnet-aux==0.0.8",
        # Utils
        "Pillow==10.3.0", "moviepy==1.0.3", "supabase==2.4.3", "reportlab==4.1.0"
    )
    .env({"HF_HOME": HF_CACHE_MOUNT})  # honor the cache mount
)

def _gpu_resource():
    # Allow CPU fallback to simplify local tests
    if GPU_TYPE.lower() == "cpu":
        return None
    try:
        return getattr(modal.gpu, GPU_TYPE)()
    except Exception:
        return modal.gpu.A10G()

# ───────────────────────────── Supabase helpers ──────────────────────────────
def _sb() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")
    return create_client(url, key)

def _public_url(path: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    return f"{base}/storage/v1/object/public/{ASSETS_BUCKET}/{path}"

def _retry(fn: Callable, attempts: int = 4, delay: float = 0.8, backoff: float = 1.8):
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(delay * (backoff ** i))
    raise last_err  # bubble up after retries

def _upload(sb: Client, key: str, data: bytes, mime: str):
    def _do():
        sb.storage.from_(ASSETS_BUCKET).upload(
            key,
            file=data,
            file_options={"contentType": mime, "upsert": "true", "cacheControl": "31536000"},
        )
    return _retry(_do)

def _download_to(sb: Client, src_key: str, dst: Path):
    def _do():
        blob = sb.storage.from_(UPLOADS_BUCKET).download(src_key)
        dst.write_bytes(blob)
    return _retry(_do)

# ───────────────────────────── Image helpers ─────────────────────────────────
def _open_downscale(p: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(p).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None
    w, h = img.size
    m = max(w, h)
    if m > MAX_EDGE:
        s = MAX_EDGE / float(m)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img

# ───────────────────── SDXL + ControlNet lazy loader ────────────────────────
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import torch

_PIPE = None
_DEVICE = None

def _select_device() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available() and GPU_TYPE.lower() != "cpu":
        return "cuda", torch.float16
    # CPU fallback
    return "cpu", torch.float32

def get_pipe(style: str = "modern"):
    """Load & memoize SDXL + depth-ControlNet with CUDA/CPU fallback."""
    global _PIPE, _DEVICE
    if _PIPE is None:
        _DEVICE, dtype = _select_device()
        print(json.dumps({"event": "pipe_init", "device": _DEVICE, "dtype": str(dtype)}))
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=dtype,
            cache_dir=HF_CACHE_MOUNT,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=dtype,
            cache_dir=HF_CACHE_MOUNT,
        )
        if _DEVICE == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.to("cuda")
        else:
            pipe.enable_attention_slicing()
            pipe.to("cpu")
        _PIPE = pipe

    style_map = {
        "modern": "modern Scandinavian furniture, light wood, airy, neutral palette",
        "industrial": "industrial loft, exposed brick, metal accents, leather",
        "contemporary": "contemporary minimalism, clean lines, designer furniture",
    }
    _PIPE.default_prompts = {
        "positive": f"photo-realistic interior, {style_map.get(style, 'modern')}, ultra high detail, 8k",
        "negative": "blurry, distorted, watermark, text, duplicate, low-quality",
    }
    return _PIPE

def _ai_stage(img: Image.Image, style: str) -> Image.Image:
    pipe = get_pipe(style)
    # torch autocast only on CUDA float16
    use_autocast = (_DEVICE == "cuda")
    kwargs = dict(
        prompt=pipe.default_prompts["positive"],
        negative_prompt=pipe.default_prompts["negative"],
        image=img,
        strength=0.45,
        guidance_scale=7.0,
        num_inference_steps=28,
    )
    if use_autocast:
        with torch.autocast("cuda"):
            out = pipe(**kwargs)
    else:
        out = pipe(**kwargs)
    return out.images[0]

# ───────────────── Flyer (PDF) & Teaser (MP4) builders ──────────────────────
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from moviepy.editor import ImageClip, concatenate_videoclips

def _build_flyer(cover_path: Path) -> Path:
    pdf = cover_path.parent / "flyer.pdf"
    c = canvas.Canvas(str(pdf), pagesize=letter)
    W, H = letter
    cover = Image.open(cover_path)
    aspect = cover.width / max(1, cover.height)
    new_h = W / max(0.01, aspect)
    c.drawImage(ImageReader(cover), 0, H - new_h, width=W, height=new_h, preserveAspectRatio=True, mask='auto')
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, H - new_h - 40, "Stage & Sell Pro — Virtual Staging")
    c.setFont("Helvetica", 12)
    c.drawString(40, H - new_h - 60, "Generated by AI • 48-hour turnaround")
    c.showPage()
    c.save()
    return pdf

def _build_teaser(staged_paths: List[Path]) -> Path:
    out = staged_paths[0].parent / "teaser.mp4"
    clips = []
    try:
        for p in staged_paths:
            clip = ImageClip(str(p)).set_duration(2.5)
            clips.append(clip)
        video = concatenate_videoclips(clips, method="compose")
        # Avoid verbose log spam and keep encode fast/compatible
        video.write_videofile(
            str(out), fps=24, codec="libx264", audio=False, preset="ultrafast", verbose=False, logger=None
        )
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        try:
            video.close()  # type: ignore
        except Exception:
            pass
    return out

def _already(job: Dict[str, Any]) -> bool:
    assets = job.get("assets") or {}
    return any(str(k).startswith("staged_") for k in assets.keys())

def _validate_job(job: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    if not isinstance(job, dict):
        return False, "invalid_job_payload"
    if "id" not in job:
        return False, "missing_job_id"
    files = job.get("files")
    if not files or not isinstance(files, list):
        return False, "no_files"
    return True, None

# ───────────────────────────── Worker functions ──────────────────────────────
@app.function(
    image=image,
    gpu=_gpu_resource(),
    timeout=1200,
    cpu=4.0,
    memory=16384,  # 16GB – SDXL w/ ControlNet is heavy; tune as needed
    network_file_systems={HF_CACHE_MOUNT: HF_CACHE},
    secrets=[modal.Secret.from_name("supabase-creds")],
)
def process_job(job: Dict[str, Any]) -> str:
    ok, reason = _validate_job(job)
    if not ok:
        print(json.dumps({"event": "reject", "reason": reason}))
        return reason or "invalid"

    if _already(job):
        print(json.dumps({"event": "skip_already", "job_id": job["id"]}))
        return f"skip {job['id']} (already delivered)"

    sb: Client = _sb()
    job_id = job["id"]
    tmp = Path(tempfile.mkdtemp(prefix=f"ssp_{job_id}_"))
    print(json.dumps({"event": "start", "job_id": job_id, "files": job.get("files", [])}))
    try:
        style = job.get("style", "m
