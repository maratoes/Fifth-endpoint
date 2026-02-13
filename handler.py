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


def initialize_model() -> LLM:
    global model
    if model is not None:
        return model
    model = LLM(
        model=os.getenv("MODEL_NAME", "browser-use/bu-30b-a3b-preview"),
        trust_remote_code=True,
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "65536")),
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
        dtype=os.getenv("DTYPE", "float16"),
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    return model


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = job.get("input", {})
        prompt = data.get("prompt", "")
        task_type = data.get("task_type", "general")
        image_b64 = data.get("image")
        if not image_b64:
            return {"error": "image is required", "status": "error"}

        image = Image.open(BytesIO(base64.b64decode(image_b64)))

        if task_type == "dom_analysis":
            system_prompt = "You are a browser automation expert. Analyze page structure and identify interactive elements."
        elif task_type == "browser_action":
            system_prompt = "You are a browser automation agent. Determine the next action to take."
        else:
            system_prompt = "You are a helpful browser assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        sampling = SamplingParams(
            max_tokens=data.get("max_tokens", 512),
            temperature=data.get("temperature", 0.6),
            top_p=data.get("top_p", 0.95),
            stop=["\n\n\n", "<|endoftext|>", "<|im_end|>"],
        )
        outputs = model.chat(messages, sampling)
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


initialize_model()
runpod.serverless.start({"handler": handler})
