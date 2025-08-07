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
        data = sb.storage.from_("uploads").download(rel)
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
        img = img.resize((int(
