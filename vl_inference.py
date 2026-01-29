"""
Vision-Language inference script using Qwen2.5-VL model.

Loads a quantized Qwen2.5-VL-3B-Instruct model and performs vision-language
inference on images with text prompts. Uses 4-bit quantization for memory efficiency.
"""

import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

# -------------------------------------------------------------------
# Environment safety (recommended)
# -------------------------------------------------------------------
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
torch.set_grad_enabled(False)

# -------------------------------------------------------------------
# Quantization config (LLM only)
# -------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "models/Qwen2.5-VL-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.eval()

# -------------------------------------------------------------------
# Processor with HARD visual token cap (CRITICAL)
# -------------------------------------------------------------------
processor = AutoProcessor.from_pretrained(
    "models/Qwen2.5-VL-3B-Instruct",
    min_pixels=256 * 28 * 28,     # ~256 tokens
    max_pixels=1024 * 28 * 28,    # upper bound
    use_fast=True,
)

# -------------------------------------------------------------------
# Input message
# -------------------------------------------------------------------
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "",
            },
            {
                "type": "text",
                "text": "Describe this image."
            },
        ],
    }
]

# -------------------------------------------------------------------
# Prepare inputs
# -------------------------------------------------------------------
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to("cuda")

# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        use_cache=True,
    )

# Trim prompt tokens
generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# Decode
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(output_text[0])