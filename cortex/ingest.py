import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from cortex.embeddings import get_embeddings


DATA_DIR = "data/documents"
PERSIST_DIR = "chroma_db"


def load_documents():
    """
    Load documents from the data directory.
    
    Supports multiple file formats:
    - PDF files (.pdf) using PyPDFLoader
    - Text files (.txt) using TextLoader
    - Word documents (.docx) using Docx2txtLoader
    
    Each document's metadata is updated with the source filename.
    
    Returns:
        list: List of Document objects loaded from the data directory.
    """
    documents = []

    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue

        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = filename
        documents.extend(loader.load())

    return documents


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
