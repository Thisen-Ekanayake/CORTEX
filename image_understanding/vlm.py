"""Qwen2.5-VL vision-language inference.

Loads a 4-bit quantized Qwen2.5-VL-3B-Instruct model lazily (on first call) and
caches it as a module-level singleton, so importing this module is cheap and the
heavy model only loads when an image is actually described.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Local model directory (override with CORTEX_VLM_MODEL).
VLM_MODEL_DIR = os.getenv("CORTEX_VLM_MODEL", "models/Qwen2.5-VL-3B-Instruct")

_model = None
_processor = None
_load_lock = threading.Lock()


def _load() -> None:
    """Load model + processor once (idempotent, thread-safe)."""
    global _model, _processor
    if _model is not None:
        return

    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.set_grad_enabled(False)

    logger.info("Loading Qwen2.5-VL from %s (4-bit)", VLM_MODEL_DIR)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    _model.eval()
    # Hard cap on visual tokens keeps memory/latency bounded.
    _processor = AutoProcessor.from_pretrained(
        VLM_MODEL_DIR,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
        use_fast=True,
    )


def describe_image(
    image_path: str,
    prompt: str = "Describe this image.",
    max_new_tokens: int = 128,
) -> str:
    """Run vision-language inference on an image with a text prompt.

    Args:
        image_path: Path to the image file.
        prompt: Instruction / question about the image.
        max_new_tokens: Generation length cap.

    Returns:
        The model's text response.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    with _load_lock:
        _load()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.inference_mode():
        generated = _model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated, strict=True)]
    decoded = _processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0].strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m image_understanding.vlm <image_path> [prompt]")
        raise SystemExit(1)
    img = sys.argv[1]
    user_prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."
    print(describe_image(img, user_prompt))
