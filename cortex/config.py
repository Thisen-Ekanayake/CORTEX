"""
Central configuration for CORTEX path constants.
Override via environment variables where noted.
"""

import os

# ── Vector store ──────────────────────────────────────────────────────────────
PERSIST_DIR: str = os.getenv("CORTEX_PERSIST_DIR", "chroma_db")

# ── Document ingestion ────────────────────────────────────────────────────────
DATA_DIR: str = os.getenv("CORTEX_DATA_DIR", "data/documents")

# ── LLM model paths ───────────────────────────────────────────────────────────
MODEL_PATH_GPU: str = os.getenv(
    "CORTEX_MODEL_GPU",
    "models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-q4_0.gguf",
)
MODEL_PATH_CPU: str = os.getenv("CORTEX_MODEL_CPU", "models/tinylama-1.1B-chat")

# ── RL feedback ───────────────────────────────────────────────────────────────
RL_FEEDBACK_DIR: str = os.getenv("CORTEX_RL_FEEDBACK_DIR", "rl_feedback")

# ── Image search cache ────────────────────────────────────────────────────────
IMAGE_CACHE_DIR: str = os.getenv("CORTEX_IMAGE_CACHE_DIR", "downloaded_images")

# ── TF-IDF classifier ─────────────────────────────────────────────────────────
TFIDF_MODEL_DIR: str = os.getenv("CORTEX_TFIDF_MODEL_DIR", "tf-idf_classifier/model")
