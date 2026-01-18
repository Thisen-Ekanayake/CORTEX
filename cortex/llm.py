import torch
from langchain_community.llms.llamacpp import LlamaCpp
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_llm(streaming=False, callbacks=None, cpu_fallback=True):
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        return LlamaCpp(
            model_path="models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-q4_0.gguf",
            n_ctx=32768,
            n_batch=64,
            max_tokens=512,
            temperature=0.2,
            n_threads=8,
            streaming=streaming,
            callbacks=callbacks or [],
            verbose=False
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