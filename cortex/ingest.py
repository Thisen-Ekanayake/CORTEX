import os
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from cortex.config import DATA_DIR, PERSIST_DIR
from cortex.embeddings import get_embeddings


def load_file(path: str) -> List[Document]:
    """
    Load a single document file.

    Supports PDF (.pdf), text (.txt), and Word (.docx). Each loaded document's
    metadata ``source`` is set to the bare filename. Unsupported extensions
    return an empty list.

    Args:
        path: Path to the file to load.

    Returns:
        List of Document objects (empty if the extension is unsupported).
    """
    filename = os.path.basename(path)

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif filename.endswith(".txt"):
        loader = TextLoader(path)
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(path)
    else:
        return []

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = filename
    return docs


def load_documents() -> List[Document]:
    """
    Load every supported document from the data directory.

    Each document's metadata is updated with the source filename.

    Returns:
        List of Document objects loaded from the data directory.
    """
    documents = []
    for filename in os.listdir(DATA_DIR):
        documents.extend(load_file(os.path.join(DATA_DIR, filename)))
    return documents


def ingest_file(path: str) -> int:
    """
    Ingest a single file into the existing Chroma vector store.

    Splits the file into chunks and adds them to the persisted collection via
    ``add_documents`` — unlike :func:`ingest`, this does not re-embed the whole
    corpus, so it is safe to call incrementally as files are uploaded.

    Args:
        path: Path to the file to ingest.

    Returns:
        Number of chunks added (0 if the file produced no loadable content).
    """
    # langchain_chroma matches the store used by the retriever in cortex.query.
    from langchain_chroma import Chroma as ChromaStore

    docs = load_file(path)
    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = ChromaStore(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )
    vectorstore.add_documents(chunks)
    return len(chunks)


def ingest():
    """
    Ingest documents into the vector database.
    
    This function performs the complete ingestion pipeline:
    1. Loads documents from the data directory
    2. Splits documents into chunks (1200 chars with 200 char overlap)
    3. Generates embeddings for each chunk
    4. Stores embeddings in ChromaDB vector store
    5. Persists the vector store to disk
    
    Prints progress messages throughout the process.
    """
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectorstore.persist()
    print("Chroma DB persisted successfully")


if __name__ == "__main__":
    ingest()
