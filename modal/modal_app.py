# modal_app.py
# ──────────────────────────────────────────────────────────────────────────────
# Stage & Sell Pro — SDXL + ControlNet virtual staging pipeline on Modal
# Features:
# - Correct SDXL + ControlNet setup (Diffusers)
# - Preloading & shared caching via Modal NetworkFileSystem (HF cache)
# - GPU acceleration (A10G by default) using the new string API (e.g. gpu="A10G")
# - Robust error handling for missing weights / OOM / CPU fallback
# - Single entrypoint `virtual_stage` returning staged JPEG bytes
# - Friendly `main` local entrypoint so `modal run modal_app.py` shows usage/help
#
# Quick usage (dev):
#   modal run modal_app.py::main                   # prints help and options
#   modal run modal_app.py::warm                   # pre-pull & warm models
#   modal run modal_app.py::_demo_virtual_stage \
#       --image /path/empty_room.jpg \
#       --room_style "modern luxury"
#
# Deployment:
#   modal deploy modal_app.py
#
# Notes:
# - Ensure you have MODAL_TOKEN_ID / MODAL_TOKEN_SECRET configured locally/CI.
# - This app uses a shared HF cache named "ssp-hf-cache" so cold-start downloads
#   happen once and are reused across workers.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import io
import os
import sys
import json
import traceback
from typing import Optional, Tuple

import modal

# ─────────────────────────── Config / Tunables ───────────────────────────────
APP_NAME = "stage-sell-pro-pipeline"

# Hugging Face cache (shared across containers)
HF_CACHE_NAME = os.getenv("HF_CACHE_NAME", "ssp-hf-cache")
HF_CACHE_MOUNT = os.getenv("HF_CACHE_MOUNT", "/root/.cache/huggingface")

# Models (override via env if desired)
SDXL_BASE = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
# Example ControlNet for SDXL (canny variant)
CONTROLNET_MODEL = os.getenv("CONTROLNET_MODEL", "diffusers/controlnet-canny-sdxl-1.0")

# Inference
DEFAULT_GUIDANCE = float(os.getenv("GUIDANCE", "5.0"))
DEFAULT_STEPS = int(os.getenv("STEPS", "28"))
MAX_EDGE = int(os.getenv("MAX_EDGE", "1536"))  # resize long edge to control VRAM
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))

# GPU selection (A10G by default) — use new string API to avoid deprecation
GPU_TYPE = os.getenv("GPU_TYPE", "A10G").upper()  # A10G, A100, H100 or "CPU"
if GPU_TYPE not in {"A10G", "A100", "H100", "CPU"}:
    GPU_TYPE = "A10G"
GPU_ARG: Optional[str] = None if GPU_TYPE == "CPU" else GPU_TYPE

# ─────────────────────────── Modal App & Image ───────────────────────────────
app = modal.App(APP_NAME)

# Shared HF cache across workers
HF_CACHE = modal.NetworkFileSystem.from_name(HF_CACHE_NAME, create_if_missing=True)

# Build image with required deps. Pin reasonable versions for stability.
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        # Core
        "torch==2.3.1",  # CUDA-enabled on GPU runtimes provided by Modal
        "transformers==4.44.0",
        "diffusers==0.31.0",
        "accelerate==0.33.0",
        "safetensors>=0.4.3",
        # Utils
        "Pillow==10.4.0",
        "opencv-python-headless==4.10.0.84",
        "numpy==2.0.1",
    )
    # Make sure HF caches to our mounted NFS path
    .env({
        "HF_HOME": HF_CACHE_MOUNT,
        "HUGGINGFACE_HUB_CACHE": HF_CACHE_MOUNT,
        "TRANSFORMERS_CACHE": HF_CACHE_MOUNT,
        "DIFFUSERS_CACHE": HF_CACHE_MOUNT,
        "PYTHONUNBUFFERED": "1",
    })
)

# ─────────────────────────── Utilities ───────────────────────────────────────

def _resize_long_edge(pil_img, max_edge: int):
    from PIL import Image

    w, h = pil_img.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return pil_img
    ratio = max_edge / float(long_edge)
    new_size = (int(w * ratio), int(h * ratio))
    return pil_img.resize(new_size, Image.LANCZOS)


def _to_jpeg_bytes(pil_img, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _canny(image_pil) -> "Image.Image":
    import numpy as np
    import cv2
    from PIL import Image

    img = np.array(image_pil.convert("RGB"))
    edges = cv2.Canny(img, 100, 200)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


# ─────────────────────────── Model Loader (Lazy) ─────────────────────────────
_pipeline = None  # global cache within container


def _load_pipeline() -> Tuple[object, str]:
    """Lazy-load SDXL + ControlNet and return (pipeline, device_desc)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline, "cached"

    import torch
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    # Attempt CUDA first if GPU is present
    use_cuda = torch.cuda.is_available() and (GPU_ARG is not None)
    dtype = torch.float16 if use_cuda else torch.float32

    # Load ControlNet and base SDXL
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

    # VAE / Memory tweaks
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_sequential_cpu_offload() if not use_cuda else pipe.enable_model_cpu_offload()
    except Exception:
        # Fallback silently if not available
        pass

    if use_cuda:
        pipe.to("cuda")
        device_desc = f"cuda:{torch.cuda.current_device()}"
    else:
        device_desc = "cpu"

    _pipeline = pipe
    return _pipeline, device_desc


# ─────────────────────────── Functions ───────────────────────────────────────

@app.function(
    image=image,
    timeout=600,
    retries=2,
    # Map the HF cache path to the shared NetworkFileSystem
    network_file_systems={HF_CACHE_MOUNT: HF_CACHE},
    gpu=GPU_ARG,  # <- use new string API (e.g. "A10G") or None for CPU
)
def warm() -> str:
    """Preload models into the shared HF cache and keep a worker warm."""
    try:
        pipe, device = _load_pipeline()
        # Compile a single dummy run to trigger weight init / graph creation
        from PIL import Image
        import torch

        dummy = Image.new("RGB", (512, 512), color=(200, 200, 200))
        canny = _canny(dummy)
        g = DEFAULT_GUIDANCE
        steps = max(10, min(DEFAULT_STEPS, 20))

        with torch.inference_mode():
            _ = pipe(
                image=canny,
                prompt="interior design style, photorealistic",
                negative_prompt="blurry, low quality, watermark",
                guidance_scale=g,
                num_inference_steps=steps,
            )
        return json.dumps({"status": "ok", "device": device, "gpu": GPU_ARG or "CPU"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "trace": traceback.format_exc()})


@app.function(
    image=image,
    timeout=900,
    retries=1,
    network_file_systems={HF_CACHE_MOUNT: HF_CACHE},
    gpu=GPU_ARG,
)
def virtual_stage(
    image: bytes,
    room_style: str = "modern luxury living room",
    negative_prompt: str = "lowres, blurry, bad anatomy, watermark, text",
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
) -> bytes:
    """
    Virtual stage an empty room photo using SDXL + ControlNet (canny guidance).

    Args:
        image: Raw bytes (JPEG/PNG) of the empty room photo.
        room_style: Prompt describing the desired staging style.
        negative_prompt: Negative prompt to avoid.
        seed: Optional RNG seed for deterministic output.
        guidance_scale: CFG guidance (default from env if None).
        num_inference_steps: Diffusion steps (default from env if None).

    Returns:
        JPEG bytes of the staged image.
    """
    from PIL import Image
    import torch

    g = guidance_scale if guidance_scale is not None else DEFAULT_GUIDANCE
    steps = num_inference_steps if num_inference_steps is not None else DEFAULT_STEPS

    try:
        # Load the pipeline (lazy)
        pipe, device = _load_pipeline()

        # Decode + resize
        inp = Image.open(io.BytesIO(image)).convert("RGB")
        inp = _resize_long_edge(inp, MAX_EDGE)

        # Build ControlNet conditioning via Canny
        control = _canny(inp)

        # Reproducibility
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=device).manual_seed(int(seed))
            except Exception:
                generator = torch.Generator().manual_seed(int(seed))

        # Inference
        with torch.inference_mode():
            result = pipe(
                image=control,
                prompt=f"{room_style}, photorealistic, high detail, interior design, furniture staging, ray-traced lighting, 8k",
                negative_prompt=negative_prompt,
                generator=generator,
                guidance_scale=g,
                num_inference_steps=steps,
            )

        # Diffusers returns a PipelineOutput with images list
        out_img = result.images[0]

        # Subtle blend with original structure (optional):
        # Keep 10% of original to reduce artifacts on walls/floor
        try:
            import numpy as np
            alpha = 0.9
            out_np = np.array(out_img).astype("float32")
            in_np = np.array(inp).astype("float32")
            blend_np = (alpha * out_np + (1 - alpha) * in_np).clip(0, 255).astype("uint8")
            out_img = Image.fromarray(blend_np)
        except Exception:
            pass

        return _to_jpeg_bytes(out_img, JPEG_QUALITY)

    except Exception as e:
        # Granular CUDA OOM handling
        err_msg = str(e)
        is_cuda_oom = any(s in err_msg.lower() for s in ["cuda out of memory", "cublas", "cudnn"]) or (
            hasattr(torch, "cuda") and torch.cuda.is_available() and isinstance(e, RuntimeError) and "out of memory" in err_msg.lower()
        )

        # Attempt a lower-res fallback if OOM
        if is_cuda_oom:
            try:
                small = _resize_long_edge(Image.open(io.BytesIO(image)).convert("RGB"), max(1024, MAX_EDGE // 2))
                control_small = _canny(small)
                pipe, device = _load_pipeline()
                with torch.inference_mode():
                    result = pipe(
                        image=control_small,
                        prompt=f"{room_style}, photorealistic, interior design, furniture staging",
                        negative_prompt=negative_prompt,
                        guidance_scale=max(4.0, g - 1.0),
                        num_inference_steps=max(18, steps - 6),
                    )
                out_img = result.images[0]
                return _to_jpeg_bytes(out_img, JPEG_QUALITY)
            except Exception:
                pass

        # If we reach here, bubble a structured error back to caller
        tb = traceback.format_exc()
        err = {
            "error": "virtual_stage_failed",
            "message": err_msg,
            "trace": tb,
            "hints": [
                "Confirm GPU is available or set GPU_TYPE=CPU for CPU fallback (slow).",
                f"Try reducing MAX_EDGE (currently {MAX_EDGE}).",
                "Lower num_inference_steps or guidance_scale.",
                "Ensure models are accessible; run warm() once to preload/cached.",
            ],
        }
        return json.dumps(err).encode("utf-8")


# ─────────────────────────── Local Entrypoints (CLI Helpers) ─────────────────

@app.local_entrypoint()
def main(command: str = "help", image: str = "", room_style: str = "modern luxury"):
    """Friendly launcher. Example:
      modal run modal_app.py::main --command warm
      modal run modal_app.py::main --command demo --image /path/img.jpg --room_style "modern luxury"
    """
    if command == "help":
        print(
            "\nStage & Sell Pro — commands:\n"
            "  • warm  → pre-download models and warm a worker\n"
            "  • demo  → run local demo on a file (writes staged_output.jpg)\n\n"
            "Examples:\n"
            "  modal run modal_app.py::warm\n"
            "  modal run modal_app.py::_demo_virtual_stage --image /path/empty_room.jpg --room_style 'modern luxury'\n"
            "  modal run modal_app.py::main --command demo --image /path/empty_room.jpg\n"
        )
        return
    if command == "warm":
        print(warm.remote())
        return
    if command == "demo":
        if not image:
            print("Missing --image /path/to/file.jpg for demo")
            sys.exit(2)
        _demo_virtual_stage.local(image=image, room_style=room_style)
        return
    print(f"Unknown command: {command}")
    sys.exit(2)


@app.local_entrypoint()
def _demo_virtual_stage(image: str, room_style: str = "modern luxury"):
    """Local convenience to test with a file path. Writes out staged_output.jpg"""
    with open(image, "rb") as f:
        data = f.read()
    out = virtual_stage.remote(data, room_style)

    # Handle structured error payloads
    try:
        # If output decodes as JSON error, print and exit
        maybe = json.loads(out.decode("utf-8"))
        if isinstance(maybe, dict) and maybe.get("error"):
            print("Error:", json.dumps(maybe, indent=2))
            sys.exit(1)
    except Exception:
        pass

    out_path = "staged_output.jpg"
    with open(out_path, "wb") as f:
        f.write(out)
    print(f"Saved → {out_path}")


@app.local_entrypoint()
def _warm():
    """Local call to warm() for cache preloading during development."""
    print(warm.remote())
