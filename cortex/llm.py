import torch
from langchain_community.llms.llamacpp import LlamaCpp
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_llm(streaming=False, callbacks=None, cpu_fallback=True):
    """
    Get or initialize the language model instance.
    
    Uses GPU if available (Qwen2.5-1.5B-Instruct via llama.cpp), otherwise
    falls back to CPU-based model (tinylama-1.1B-chat) if cpu_fallback is True.
    
    Args:
        streaming: Whether to enable streaming mode for token-by-token generation.
        callbacks: List of callback handlers for streaming/events.
        cpu_fallback: If True, use CPU model when GPU is unavailable (default: True).
    
    Returns:
        LLM instance: Either LlamaCpp (GPU) or pipeline (CPU) model.
    
    Raises:
        RuntimeError: If GPU unavailable and cpu_fallback is False.
    """
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        return LlamaCpp(
            model_path="models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-q4_0.gguf",
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
            stop=["</s>", "<|eot_id|>", "<|endoftext|>", "\nUser:", "\nAssistant:"]
        )
    
    elif cpu_fallback:
        model_name = "tinylama-1.1B-chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    
    else:
        raise RuntimeError("No GPU detected and CPU fallback disabled")