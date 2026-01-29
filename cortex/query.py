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
# RAG chain
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
# RAG execution
# -------------------------
def run_rag(query: str, callbacks=None) -> Optional[str]:
    """
    Executes RAG only if relevant documents exist.
    Returns None if no relevant docs found.
    """
    return run_rag_mode(query=query, mode="document", callbacks=callbacks)


def run_rag_mode(
    query: str,
    mode: Literal["document", "image"] = "document",
    callbacks=None,
) -> Optional[str]:
    """
    Execute RAG using either:
    - mode="document": vector DB retrieval + LLM answer grounded in text context
    - mode="image": image retrieval (best-effort) returning top matches

    Returns None when no relevant results are available for the chosen mode.
    """
    if mode == "image":
        # Best-effort image retrieval integration. This returns a textual list of
        # retrieved images (paths) and similarity scores.
        try:
            from image_search.realtime_retrieval import search_and_retrieve
        except Exception:
            return None

        try:
            results = search_and_retrieve(query, num_images=10, top_k=5)
        except Exception:
            return None

        if not results:
            return None

        lines = ["Top image matches:"]
        for path, score in results:
            lines.append(f"- {path} (score: {score:.3f})")
        return "\n".join(lines)

    # Default: document RAG
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


def get_sources(query: str) -> List[str]:
    """
    Get source citations for retrieved documents.
    """
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
    """
    Answer questions about the system using persona information.
    Generates natural responses based on CORTEX_SYSTEM_PROMPT.
    """
    llm = get_llm(streaming=True, callbacks=callbacks)
    
    meta_prompt = f"""Based on this system information:

{CORTEX_SYSTEM_PROMPT}

Answer the user's question naturally and conversationally.
Don't just repeat the information verbatim - explain it in a friendly, helpful way.
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
    """
    Handle general conversation without document retrieval.
    """
    llm = get_llm(streaming=True, callbacks=callbacks)
    
    # Include system prompt for context about the assistant's identity
    chat_prompt = f"""{CORTEX_SYSTEM_PROMPT}

User: {query}
Assistant:"""
    
    if callbacks:
        chunks = []
        for chunk in llm.stream(chat_prompt):
            chunks.append(chunk)
        return "".join(chunks)
    
    return llm.invoke(chat_prompt)