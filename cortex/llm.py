from langchain_community.llms import LlamaCpp


def get_llm():
    return LlamaCpp(
        model_path="models/qwen2.5-1.5b-instruct-q4_0.gguf",
        n_ctx=2048,
        max_tokens=512,
        temperature=0.2,
        n_threads=8,
        verbose=False
    )
