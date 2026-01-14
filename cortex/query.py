from typing import List, Tuple, Optional

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from cortex.embeddings import get_embeddings
from cortex.llm import get_llm

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

    # Create a runnable that retrieves and formats documents
    # Use retrieve_docs which now uses vectorstore directly
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
# Public API
# -------------------------
def run_rag(query: str, callbacks=None) -> Optional[str]:
    """
    Executes RAG only if relevant documents exist.
    Returns None if RAG should NOT be used.
    """
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
    Explicit source retrieval (only call after RAG ran).
    """
    docs = retrieve_docs(query)
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        sources.append(f"{source}, page {page}")

    return sources
