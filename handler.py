import base64
import logging
import os
from io import BytesIO
from typing import Any, Dict

import runpod
from PIL import Image
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = None


def _configure_cache_dirs() -> None:
    """Prefer caching to a mounted network volume when available."""
    volume_root = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")
    if not os.path.isdir(volume_root):
        return

    cache_root = os.path.join(volume_root, "cache")
    hf_home = os.getenv("HF_HOME") or os.path.join(cache_root, "hf")
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE") or os.path.join(hf_home, "hub")
    vllm_cache = os.getenv("VLLM_CACHE_ROOT") or os.path.join(cache_root, "vllm")

    os.makedirs(hub_cache, exist_ok=True)
    os.makedirs(vllm_cache, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub_cache)
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("VLLM_CACHE_ROOT", vllm_cache)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def initialize_model() -> LLM:
    global model
    if model is not None:
        return model
    _configure_cache_dirs()
    enforce_eager = os.getenv("ENFORCE_EAGER", "1").strip().lower() in {"1", "true", "yes", "y"}
    enable_prefix_caching = os.getenv("ENABLE_PREFIX_CACHING", "0").strip().lower() in {"1", "true", "yes", "y"}
    model = LLM(
        model=os.getenv("MODEL_NAME", "browser-use/bu-30b-a3b-preview"),
        trust_remote_code=True,
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "65536")),
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
        dtype=os.getenv("DTYPE", "float16"),
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
        limit_mm_per_prompt={"image": 1},
    )
    return model


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if model is None:
            try:
                initialize_model()
            except Exception as exc:  # noqa: BLE001
                return {"error": f"model_init_failed: {exc}", "status": "error"}

        data = job.get("input", {})
        prompt = data.get("prompt", "")
        task_type = data.get("task_type", "general")
        image_b64 = data.get("image")
        if not image_b64:
            return {"error": "image is required", "status": "error"}

        if task_type == "dom_analysis":
            system_prompt = "You are a browser automation expert. Analyze page structure and identify interactive elements."
        elif task_type == "browser_action":
            system_prompt = "You are a browser automation agent. Determine the next action to take."
        else:
            system_prompt = "You are a helpful browser assistant."

        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        # Force full decode early; avoid downstream "broken data stream" errors.
        image.load()
        image = image.convert("RGB")
        image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Picture 1: {image_placeholder}{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        sampling = SamplingParams(
            max_tokens=data.get("max_tokens", 512),
            temperature=data.get("temperature", 0.6),
            top_p=data.get("top_p", 0.95),
            stop=["\n\n\n", "<|endoftext|>", "<|im_end|>"],
        )
        outputs = model.generate(
            [{"prompt": full_prompt, "multi_modal_data": {"image": image}}],
            sampling,
        )
        result = outputs[0].outputs[0].text
        return {
            "output": result,
            "status": "success",
            "task_type": task_type,
            "model": "browser-use/bu-30b-a3b-preview",
        }
    except Exception as exc:
        logger.exception("handler error")
        return {"error": str(exc), "status": "error"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
