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
