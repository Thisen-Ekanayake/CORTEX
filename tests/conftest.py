"""
Bootstrap: stub all heavy ML/UI packages before project code is imported.

The test suite intentionally avoids installing torch, llama-cpp, chromadb, and
the full langchain/HuggingFace stack. This conftest inserts lightweight stubs
into sys.modules so those imports resolve without error. Tests that exercise
logic which calls through to the real ML stack patch at the call site instead.

Minimum test dependencies:
    pip install pytest pytest-cov scikit-learn joblib numpy
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock


# ── Helper ────────────────────────────────────────────────────────────────────

def _stub(name: str) -> MagicMock:
    """Insert a MagicMock into sys.modules under *name* (no-op if already present)."""
    if name not in sys.modules:
        m = MagicMock()
        m.__name__ = name
        sys.modules[name] = m
    return sys.modules[name]  # type: ignore[return-value]


# ── torch ─────────────────────────────────────────────────────────────────────
# Build the torch stub manually so torch.cuda.is_available() returns False.
_torch_cuda = MagicMock()
_torch_cuda.is_available = MagicMock(return_value=False)

_torch = MagicMock()
_torch.__name__ = "torch"
_torch.cuda = _torch_cuda
sys.modules.setdefault("torch", _torch)


# ── langchain_core ────────────────────────────────────────────────────────────
# BaseCallbackHandler must be a real class so StreamHandler can inherit from it.

class _BaseCallbackHandler:
    pass


class _Document:
    """Minimal Document stub used in ingest.py / query.py type hints."""

    def __init__(self, page_content: str = "", metadata: dict | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


_lc_callbacks = _stub("langchain_core.callbacks")
_lc_callbacks.BaseCallbackHandler = _BaseCallbackHandler

_lc_docs = _stub("langchain_core.documents")
_lc_docs.Document = _Document

for _mod in [
    "langchain_core",
    "langchain_core.output_parsers",
    "langchain_core.prompts",
    "langchain_core.runnables",
]:
    _stub(_mod)


# ── rest of the langchain / vector-store stack ────────────────────────────────
for _mod in [
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_community.vectorstores",
    "langchain_community.llms",
    "langchain_community.llms.llamacpp",
    "langchain_chroma",
    "langchain_huggingface",
    "langchain_text_splitters",
    "chromadb",
    "sentence_transformers",
    "transformers",
]:
    _stub(_mod)


# ── other heavy / optional packages ──────────────────────────────────────────
for _mod in [
    "wandb",
    "sounddevice",
    "pydub",
    "pynput",
    "TTS",
    "TTS.api",
    # image_search submodules — stubbing these prevents google_search.py
    # from executing its module-level os.environ["GOOGLE_API_KEY"] read.
    # image_search itself is a real package (has __init__.py) so we do NOT
    # stub it here — Python must be able to import realtime_retrieval.py from
    # disk to test its download_images logic.
    "google_search",
    "clip_retrieval_local",
    "image_search.google_search",
    "image_search.clip_retrieval_local",
    # module-level env-var reads inside google_search.py are bypassed by stubs
    "dotenv",
    "google_images_search",
    # UI (Rich / prompt_toolkit) – not needed for unit tests
    "rich",
    "rich.console",
    "rich.panel",
    "rich.text",
    "rich.live",
    "rich.spinner",
    "rich.status",
    "rich.progress",
    "rich.markdown",
    "rich.table",
    "rich.align",
    "prompt_toolkit",
    "prompt_toolkit.history",
    "prompt_toolkit.auto_suggest",
    "prompt_toolkit.completion",
    "prompt_toolkit.styles",
    "prompt_toolkit.formatted_text",
]:
    _stub(_mod)
