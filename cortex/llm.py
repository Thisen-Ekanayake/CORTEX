import logging
from typing import Any, List, Optional, Union

import torch
from langchain_community.llms.llamacpp import LlamaCpp
from transformers import pipeline

from cortex.config import MODEL_PATH_CPU, MODEL_PATH_GPU

logger = logging.getLogger(__name__)


def get_llm(
    streaming: bool = False,
    callbacks: Optional[List[Any]] = None,
    cpu_fallback: bool = True,
) -> Union[LlamaCpp, Any]:
    """
    Get or initialize the language model instance.

    Uses GPU if available (Qwen2.5-1.5B-Instruct via llama.cpp), otherwise
    falls back to CPU-based model (tinylama-1.1B-chat) if cpu_fallback is True.

    Args:
        streaming: Whether to enable streaming mode for token-by-token generation.
        callbacks: List of callback handlers for streaming/events.
        cpu_fallback: If True, use CPU model when GPU is unavailable (default: True).

    Returns:
        LLM instance: Either LlamaCpp (GPU) or HF pipeline (CPU).

    Raises:
        RuntimeError: If GPU unavailable and cpu_fallback is False.
    """
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        return LlamaCpp(
            model_path=MODEL_PATH_GPU,
            n_ctx=32768,
            n_batch=64,
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.15,
            n_threads=8,
            streaming=streaming,
            callbacks=callbacks or [],
            verbose=False,
            stop=["</s>", "<|eot_id|>", "<|endoftext|>", "\nUser:", "\nAssistant:"],
        )

    if cpu_fallback:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("GPU unavailable — loading CPU fallback model: %s", MODEL_PATH_CPU)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_CPU)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_CPU)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    raise RuntimeError("No GPU detected and CPU fallback disabled")
