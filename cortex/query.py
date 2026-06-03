import logging
from typing import Any, List, Optional, Literal

from langchain_chroma import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from cortex.config import PERSIST_DIR
from cortex.embeddings import get_embeddings
from cortex.llm import get_llm
from cortex.persona import CORTEX_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# -------------------------
# Internal caches
# -------------------------
_vectorstore = None
_retriever = None
_rag_chain = None


# -------------------------
# Vector store / retriever
# -------------------------
def get_retriever():
    global _vectorstore, _retriever

    if _retriever is not None:
        return _retriever

    embeddings = get_embeddings()

    _vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    _retriever = _vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    return _retriever


# -------------------------
# Document utilities
# -------------------------
def retrieve_docs(query: str):
    retriever = get_retriever()
    return retriever.invoke(query)


def format_docs(docs) -> str:
    if not docs:
        return ""

    chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        chunks.append(f"[{source}, page {page}]\n{content}")

    return "\n\n".join(chunks)


# -------------------------
# RAG chain (documents)
# -------------------------
def get_rag_chain(callbacks: Optional[List[Any]] = None):
    global _rag_chain

    if _rag_chain and not callbacks:
        return _rag_chain

    llm = get_llm(streaming=True, callbacks=callbacks)

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.
If the context does not contain the answer, respond with "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    def retrieve_and_format(query: str) -> str:
        docs = retrieve_docs(query)
        return format_docs(docs)

    chain = (
        {
            "context": RunnableLambda(retrieve_and_format),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if not callbacks:
        _rag_chain = chain

    return chain


# -------------------------
# Unified RAG execution
# -------------------------
def run_rag_mode(
    query: str,
    mode: Literal["document", "image"] = "document",
    callbacks: Optional[List[Any]] = None,
) -> Optional[str]:
    """
    mode="document" → text RAG using vector DB
    mode="image"    → image retrieval via Google + CLIP
    """

    # ---------- IMAGE RAG ----------
    if mode == "image":
        try:
            from image_search.realtime_retrieval import search_and_retrieve
        except ImportError as exc:
            logger.warning("image_search module unavailable: %s", exc)
            return None

        try:
            results = search_and_retrieve(
                query,
                num_images=10,
                top_k=5
            )
        except Exception as exc:
            logger.error("Image retrieval failed for query %r: %s", query, exc)
            return None

        if not results:
            return None

        lines = ["Top image matches:"]
        for path, score in results:
            lines.append(f"- {path} (score: {score:.3f})")

        return "\n".join(lines)

    # ---------- DOCUMENT RAG ----------
    docs = retrieve_docs(query)
    if not docs:
        return None

    chain = get_rag_chain(callbacks=callbacks)

    if callbacks:
        chunks = []
        for chunk in chain.stream(query):
            chunks.append(chunk)
        return "".join(chunks)

    return chain.invoke(query)


# -------------------------
# Convenience wrapper
# -------------------------
def run_rag(query: str, callbacks: Optional[List[Any]] = None) -> Optional[str]:
    return run_rag_mode(
        query=query,
        mode="document",
        callbacks=callbacks
    )


# -------------------------
# Scored semantic search
# -------------------------
def search_with_scores(query: str, k: int = 5) -> List[dict]:
    """
    Semantic search returning ranked chunks with relevance scores.

    Unlike the retriever used by the RAG chain (which hides scores), this
    exposes the similarity score for each hit so the UI can rank/visualize
    results.

    Args:
        query: Search string.
        k: Maximum number of hits to return.

    Returns:
        List of dicts: {source, page, snippet, score} ordered by relevance.
    """
    get_retriever()  # ensures the module-level vectorstore is initialized
    results = _vectorstore.similarity_search_with_relevance_scores(query, k=k)

    hits = []
    for doc, score in results:
        hits.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": str(doc.metadata.get("page", "")),
                "snippet": doc.page_content.strip()[:300],
                "score": float(score),
            }
        )
    return hits


# -------------------------
# Source citations
# -------------------------
def get_sources(query: str) -> List[str]:
    docs = retrieve_docs(query)
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        sources.append(f"{source}, page {page}")

    return sources


# -------------------------
# META execution
# -------------------------
def run_meta(query: str, callbacks: Optional[List[Any]] = None) -> str:
    llm = get_llm(streaming=True, callbacks=callbacks)

    meta_prompt = f"""Based on this system information:

{CORTEX_SYSTEM_PROMPT}

Answer the user's question naturally and conversationally.
Be concise but informative.

User: {query}
Assistant:"""

    if callbacks:
        chunks = []
        for chunk in llm.stream(meta_prompt):
            chunks.append(chunk)
        return "".join(chunks)

    return llm.invoke(meta_prompt)


# -------------------------
# CHAT execution
# -------------------------
def run_chat(query: str, callbacks: Optional[List[Any]] = None) -> str:
    llm = get_llm(streaming=True, callbacks=callbacks)

    chat_prompt = f"""{CORTEX_SYSTEM_PROMPT}

User: {query}
Assistant:"""

    if callbacks:
        chunks = []
        for chunk in llm.stream(chat_prompt):
            chunks.append(chunk)
        return "".join(chunks)

    return llm.invoke(chat_prompt)
