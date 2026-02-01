from typing import List, Optional, Literal

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from cortex.embeddings import get_embeddings
from cortex.llm import get_llm
from cortex.persona import CORTEX_SYSTEM_PROMPT

PERSIST_DIR = "chroma_db"

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
def get_rag_chain(callbacks=None):
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
    callbacks=None,
) -> Optional[str]:
    """
    mode="document" → text RAG using vector DB
    mode="image"    → image retrieval via Google + CLIP
    """

    # ---------- IMAGE RAG ----------
    if mode == "image":
        try:
            from image_search.realtime_retrieval import search_and_retrieve
        except Exception:
            return None

        try:
            results = search_and_retrieve(
                query,
                num_images=10,
                top_k=5
            )
        except Exception:
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
def run_rag(query: str, callbacks=None) -> Optional[str]:
    return run_rag_mode(
        query=query,
        mode="document",
        callbacks=callbacks
    )


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
def run_meta(query: str, callbacks=None) -> str:
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
def run_chat(query: str, callbacks=None) -> str:
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
