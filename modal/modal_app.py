import os
from pathlib import Path
import tempfile, shutil, io
from typing import Dict, Any, List, Optional

import modal
from supabase import create_client, Client
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError

# ──────────────────────────────────────────────────────────────────────────────
# App: Stage & Sell Pro automated pipeline
# ──────────────────────────────────────────────────────────────────────────────
app = modal.App("stage-sell-pro-pipeline")

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("Pillow==10.3.0", "moviepy==1.0.3", "supabase==2.4.3")
)

# Safety knobs
MAX_EDGE = 2048            # downscale long edge to avoid huge files / OOM
JPEG_QUALITY = 92

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _sb() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]  # service role in Modal secret
    return create_client(url, key)

def _public_url(path: str) -> str:
    return f"{os.environ['SUPABASE_URL'].rstrip('/')}/storage/v1/object/public/assets/{path}"

def _download_uploads(sb: Client, files: List[str], tmpdir: Path) -> List[Path]:
    out: List[Path] = []
    for rel in files:
        data = sb.storage.from_("uploads").download(rel)              # supabase-py v2 bytes
        dst = tmpdir / Path(rel).name
        with open(dst, "wb") as f:
            f.write(data)
        out.append(dst)
    return out

def _open_and_downscale(p: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(p).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None
    w, h = img.size
    m = max(w, h)
    if m > MAX_EDGE:
        scale = MAX_EDGE / float(m)
        img = img.resize(
            (int(w * scale), int(h * scale)),   # ★ fixed parentheses
            Image.LANCZOS
        )
    return img

def _simple_stage(img: Image.Image) -> Image.Image:
    # Placeholder: minimal enhancement
    img = ImageEnhance.Brightness(img).enhance(1.05)
    img = ImageEnhance.Color(img).enhance(1.08)
    img = img.filter(ImageFilter.SHARPEN)
    return img

def _upload_asset(sb: Client, key: str, pil_img: Image.Image):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    sb.storage.from_("assets").upload(
        key,
        file=buf.getvalue(),
        file_options={"contentType": "image/jpeg", "upsert": "true", "cacheControl": "31536000"}
    )

def _already_delivered(job: Dict[str, Any]) -> bool:
    return any(k.startswith("staged_") for k in (job.get("assets") or {}).keys())

# ──────────────────────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────────────────────
@app.function(image=image, timeout=900, secrets=[modal.Secret.from_name("supabase-creds")])
def process_job(job: Dict[str, Any]) -> str:
    sb = _sb()
    job_id = job["id"]
    files = job.get("files") or []

    if _already_delivered(job):
        return f"skip (already delivered) {job_id}"

    if not files:
        sb.table("jobs").update({"status": "error", "assets": {"reason": "no_files"}}).eq("id", job_id).execute()
        return "no-files"

    tmp = Path(tempfile.mkdtemp())
    created: Dict[str, str] = {}
    try:
        srcs = _download_uploads(sb, files, tmp)
        idx = 0
        for src in srcs:
            img = _open_and_downscale(src)
            if img is None:
                continue
            try:
                staged = _simple_stage(img)
                idx += 1
                key = f"{job_id}/staged_{idx:02d}.jpg"
                _upload_asset(sb, key, staged)
                created[f"staged_{idx:02d}"] = _public_url(key)
            except Exception as e:
                print(f"[warn] failed staging {src.name}: {e}")

        if not created:
            sb.table("jobs").update({"status": "error", "assets": {"reason": "no_valid_images"}}).eq("id", job_id).execute()
            return "no-valid-images"

        created.setdefault("flyer_cover", next(iter(created.values())))
        sb.table("jobs").update({"status": "delivered", "assets": created}).eq("id", job_id).execute()
        return f"delivered {job_id} ({len([k for k in created if k.startswith('staged_')])} images)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@app.function(
    image=image,
    schedule=modal.Period(seconds=120),         # every 2 minutes
    secrets=[modal.Secret.from_name("supabase-creds")]
)
def poll_and_process() -> str:
    sb = _sb()
    resp = sb.table("jobs").select("*").eq("status", "queued").order("created_at", desc=False).limit(1).execute()
    rows = resp.data or []
    if not rows:
        return "no-queued-jobs"
    job = rows[0]
    sb.table("jobs").update({"status": "processing"}).eq("id", job["id"]).execute()
    process_job.spawn(job)
    return f"spawned {job['id']}"

# Local test (optional)
@app.local_entrypoint()
def run_local_test():
    job = {"id": "test-job", "files": []}
    print(process_job.fn(job))
