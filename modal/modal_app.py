import os
from pathlib import Path
import tempfile, shutil, io
from typing import Dict, Any, List

import modal
from supabase import create_client, Client
from PIL import Image, ImageEnhance, ImageFilter

# App: Stage & Sell Pro automated pipeline
app = modal.App("stage-sell-pro-pipeline")

# Use the modern base image helper (no DockerHub dependency)
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("Pillow==10.3.0", "moviepy==1.0.3", "supabase==2.4.3")
)

# Optional persistent cache for future ML models
CACHE = modal.SharedVolume().persist("ssp-cache")

def _sb() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)

def _public_url(path: str) -> str:
    # Build a public URL for the assets bucket
    return f"{os.environ['SUPABASE_URL'].rstrip('/')}/storage/v1/object/public/assets/{path}"

def _download_uploads(sb: Client, files: List[str], tmpdir: Path) -> List[Path]:
    out: List[Path] = []
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
    # NOTE: use camelCase file_options with supabase-py v2
    sb.storage.from_("assets").upload(
        key,
        buf.getvalue(),
        file_options={"contentType": "image/jpeg", "upsert": "true", "cacheControl": "31536000"}
    )

@app.function(image=image, timeout=900, secrets=[modal.Secret.from_name("supabase-creds")])
def process_job(job: Dict[str, Any]):
    sb = _sb()
    job_id = job["id"]
    files = job.get("files") or []
    if not files:
        sb.table("jobs").update({"status": "error", "assets": {"reason": "no_files"}}).eq("id", job_id).execute()
        return "no-files"

    tmp = Path(tempfile.mkdtemp())
    assets_map: Dict[str, str] = {}
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
        if assets_map and "flyer_cover" not in assets_map:
            assets_map["flyer_cover"] = next(iter(assets_map.values()))

        # Update DB
        sb.table("jobs").update({"status": "delivered", "assets": assets_map}).eq("id", job_id).execute()
        return f"delivered {job_id}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@app.function(
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
