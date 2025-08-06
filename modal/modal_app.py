import os
from pathlib import Path
import tempfile, shutil, io
from typing import Dict, Any, List

import modal
from supabase import create_client, Client
from PIL import Image, ImageEnhance, ImageFilter

# App: Stage & Sell Pro automated pipeline
stub = modal.Stub("stage-sell-pro-pipeline")

image = (
    modal.Image.from_dockerhub("python:3.11-slim")
    .apt_install("ffmpeg")
    .pip_install("Pillow==10.3.0", "moviepy==1.0.3", "supabase==2.4.3")
)

# Optional persistent HF cache for future ML models (not used in this stub)
CACHE = modal.SharedVolume().persist("ssp-cache")

def _sb() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)

def _public_url(path: str) -> str:
    # Build a public URL for the assets bucket
    return f"{os.environ['SUPABASE_URL'].rstrip('/')}/storage/v1/object/public/assets/{path}"

def _download_uploads(sb: Client, files: List[str], tmpdir: Path) -> List[Path]:
    out = []
    for rel in files:
        # download bytes from 'uploads' bucket
        data = sb.storage.from_("uploads").download(rel)
        fn = Path(rel).name
        p = tmpdir / fn
        with open(p, "wb") as f:
            f.write(data)
        out.append(p)
    return out

def _simple_stage(img_path: Path) -> Image.Image:
    # Placeholder: subtle enhancements to demonstrate pipeline.
    im = Image.open(img_path).convert("RGB")
    im = ImageEnhance.Brightness(im).enhance(1.05)
    im = ImageEnhance.Color(im).enhance(1.08)
    im = im.filter(ImageFilter.SHARPEN)
    return im

def _upload_asset(sb: Client, key: str, pil_img: Image.Image):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    sb.storage.from_("assets").upload(key, buf.getvalue(), {"content-type": "image/jpeg", "upsert": "true"})

@stub.function(image=image, timeout=900, secrets=[modal.Secret.from_name("supabase-creds")])
def process_job(job: Dict[str, Any]):
    sb = _sb()
    job_id = job["id"]
    files = job.get("files") or []
    tmp = Path(tempfile.mkdtemp())
    assets_map = {}
    try:
        # Download sources
        local_paths = _download_uploads(sb, files, tmp)
        # Process each image (stub enhancement)
        for i, src_path in enumerate(local_paths, 1):
            staged = _simple_stage(src_path)
            key = f"{job_id}/staged_{i:02d}.jpg"
            _upload_asset(sb, key, staged)
            assets_map[f"staged_{i:02d}"] = _public_url(key)

        # Demo flyer placeholder: just reuse first staged as cover
        if assets_map:
            assets_map["flyer_cover"] = list(assets_map.values())[0]

        # Update DB
        sb.table("jobs").update({"status": "delivered", "assets": assets_map}).eq("id", job_id).execute()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@stub.function(
    image=image,
    schedule=modal.Periodic("*/2 * * * *"),  # every 2 minutes
    secrets=[modal.Secret.from_name("supabase-creds")]
)
def poll_and_process():
    sb = _sb()
    resp = sb.table("jobs").select("*").eq("status", "queued").order("created_at", desc=False).limit(1).execute()
    rows = resp.data or []
    if not rows:
        return "no-queued-jobs"
    job = rows[0]
    # move to processing
    sb.table("jobs").update({"status": "processing"}).eq("id", job["id"]).execute()
    process_job.spawn(job)
    return f"spawned {job['id']}"
