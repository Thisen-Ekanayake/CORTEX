# ingest.py Documentation

## Overview

This module handles the document ingestion pipeline for a RAG (Retrieval-Augmented Generation) system. It loads documents from various formats, splits them into manageable chunks, generates embeddings, and stores them in a Chroma vector database for efficient semantic search and retrieval.

## Pipeline Architecture

```
Documents (PDF/TXT/DOCX) 
    ↓
Load Documents
    ↓
Split into Chunks (1200 chars, 200 overlap)
    ↓
Generate Embeddings (all-MiniLM-L6-v2)
    ↓
Store in Chroma Vector Database
    ↓
Persist to Disk
```

## Core Functions

### `load_documents()`

Loads all supported documents from the data directory.

**Supported Formats:**
- PDF files (`.pdf`)
- Text files (`.txt`)
- Word documents (`.docx`)

**Returns:** List of LangChain `Document` objects with metadata

**Process:**
1. Scans the `DATA_DIR` directory
2. Identifies files by extension
3. Uses appropriate loader for each file type
4. Adds source filename to metadata
5. Returns combined document list

**Example Output:**
```python
[
    Document(
        page_content="Document text here...",
        metadata={'source': 'example.pdf', 'page': 0}
    ),
    ...
]
```

### `ingest()`

Main orchestration function that runs the complete ingestion pipeline.

**Steps:**
1. Load all documents from `DATA_DIR`
2. Split documents into chunks
3. Generate embeddings for each chunk
4. Create/update Chroma vector store
5. Persist to disk

**Output:** Prints progress information to console

## Configuration

### Directory Configuration

```python
DATA_DIR = "data/documents"      # Source documents location
PERSIST_DIR = "chroma_db"        # Vector database storage location
```

#### Customization Options:

```python
# Use absolute paths
DATA_DIR = "/path/to/your/documents"
PERSIST_DIR = "/path/to/vector/store"

# Multiple source directories (requires code modification)
DATA_DIRS = ["data/pdfs", "data/txts", "data/docx"]

# Environment variables
import os
DATA_DIR = os.getenv("DOCUMENT_DIR", "data/documents")
PERSIST_DIR = os.getenv("CHROMA_DIR", "chroma_db")
```

### Text Splitting Configuration

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,        # Characters per chunk
    chunk_overlap=200       # Overlapping characters
)
```

## Hyperparameters Guide

### 1. Chunk Size (`chunk_size`)

**Current Value:** 1200 characters

**What it does:** Determines the maximum size of each text chunk

**Tuning Guidelines:**

| Chunk Size | Use Case | Pros | Cons |
|------------|----------|------|------|
| 200-500 | Precise retrieval, Q&A | Fine-grained search | May lose context |
| 500-1000 | Balanced approach | Good context preservation | Moderate retrieval precision |
| 1000-2000 | Long-form content | Rich context | Less precise retrieval |
| 2000+ | Summarization tasks | Maximum context | Poor retrieval accuracy |

**Recommendations:**
- **Q&A Systems:** 300-800 characters
- **General RAG:** 800-1500 characters
- **Code Documentation:** 500-1000 characters
- **Legal/Medical Docs:** 1500-2500 characters

```python
# For Q&A systems
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# For long-form content
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300
)
```

### 2. Chunk Overlap (`chunk_overlap`)

**Current Value:** 200 characters

**What it does:** Creates overlapping content between consecutive chunks to preserve context across boundaries

**Tuning Guidelines:**
- **General Rule:** 10-20% of chunk_size
- **Minimum:** 50 characters
- **Maximum:** 50% of chunk_size

**Recommendations by Chunk Size:**
- chunk_size=500 → overlap=50-100
- chunk_size=1000 → overlap=100-200
- chunk_size=1200 → overlap=150-250
- chunk_size=2000 → overlap=300-400

```python
# Calculate optimal overlap
chunk_size = 1200
chunk_overlap = int(chunk_size * 0.15)  # 15% overlap = 180
```

### 3. Separators (Advanced)

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Default order
)
```

**Custom separators for specific content:**

```python
# For code
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
)

# For structured documents
structured_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
)
```

## Alternative Document Loaders

### Currently Supported Loaders

#### 1. PyPDFLoader
```python
loader = PyPDFLoader(path)
```
- **Best for:** Standard PDFs
- **Features:** Page-by-page extraction
- **Limitations:** May struggle with complex layouts

**Alternatives:**
```python
# Better OCR and table extraction
from langchain_community.document_loaders import PDFPlumberLoader
loader = PDFPlumberLoader(path)

# For scanned PDFs (OCR)
from langchain_community.document_loaders import PDFMinerLoader
loader = PDFMinerLoader(path)

# Advanced PDF parsing
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader(path)

# Unstructured (handles complex layouts)
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader(path)
```

#### 2. TextLoader
```python
loader = TextLoader(path)
```
- **Best for:** Plain text files
- **Encoding:** UTF-8 (default)

**With custom encoding:**
```python
loader = TextLoader(path, encoding='latin-1')
```

#### 3. Docx2txtLoader
```python
loader = Docx2txtLoader(path)
```
- **Best for:** Microsoft Word documents

**Alternative:**
```python
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
loader = UnstructuredWordDocumentLoader(path)
```

### Additional Loaders to Consider

#### For Markdown Files
```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

if filename.endswith(".md"):
    loader = UnstructuredMarkdownLoader(path)
```

#### For CSV Files
```python
from langchain_community.document_loaders import CSVLoader

if filename.endswith(".csv"):
    loader = CSVLoader(path)
```

#### For HTML/Web Pages
```python
from langchain_community.document_loaders import UnstructuredHTMLLoader

if filename.endswith(".html"):
    loader = UnstructuredHTMLLoader(path)
```

#### For JSON Files
```python
from langchain_community.document_loaders import JSONLoader

if filename.endswith(".json"):
    loader = JSONLoader(
        file_path=path,
        jq_schema='.content',  # JSONPath to text content
        text_content=False
    )
```

#### For Excel Files
```python
from langchain_community.document_loaders import UnstructuredExcelLoader

if filename.endswith((".xlsx", ".xls")):
    loader = UnstructuredExcelLoader(path)
```

#### For Images (OCR)
```python
from langchain_community.document_loaders import UnstructuredImageLoader

if filename.endswith((".png", ".jpg", ".jpeg")):
    loader = UnstructuredImageLoader(path)
```

## Enhanced `load_documents()` Function

### With Additional Format Support

```python
def load_documents():
    from langchain_community.document_loaders import (
        PyPDFLoader, TextLoader, Docx2txtLoader,
        UnstructuredMarkdownLoader, CSVLoader,
        UnstructuredHTMLLoader, JSONLoader
    )
    
    documents = []
    
    loader_map = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.md': UnstructuredMarkdownLoader,
        '.csv': CSVLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
    }
    
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        
        if not os.path.isfile(path):
            continue
            
        ext = os.path.splitext(filename)[1].lower()
        loader_class = loader_map.get(ext)
        
        if loader_class is None:
            print(f"Skipping unsupported file: {filename}")
            continue
        
        try:
            loader = loader_class(path)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = filename
                
            documents.extend(docs)
            print(f"Loaded: {filename} ({len(docs)} documents)")
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    return documents
```

### With Error Handling and Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents():
    documents = []
    successful = 0
    failed = 0
    
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif filename.endswith(".txt"):
                loader = TextLoader(path, encoding='utf-8')
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                continue
            
            docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = filename
                doc.metadata['file_path'] = path
                doc.metadata['file_type'] = os.path.splitext(filename)[1]
            
            documents.extend(docs)
            successful += 1
            logger.info(f"✓ Loaded {filename}: {len(docs)} documents")
            
        except Exception as e:
            failed += 1
            logger.error(f"✗ Failed to load {filename}: {str(e)}")
            continue
    
    logger.info(f"Summary: {successful} successful, {failed} failed")
    return documents
```

## Alternative Text Splitters

### 1. CharacterTextSplitter
```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)
```
- **Best for:** Simple splitting by separator
- **Limitation:** Less flexible than RecursiveCharacterTextSplitter

### 2. TokenTextSplitter
```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=300,      # In tokens, not characters
    chunk_overlap=50
)
```
- **Best for:** Respecting LLM token limits
- **Use case:** Ensuring chunks fit model context windows

### 3. MarkdownHeaderTextSplitter
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
```
- **Best for:** Markdown documents
- **Feature:** Preserves document structure

### 4. HTMLHeaderTextSplitter
```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
```
- **Best for:** HTML documents
- **Feature:** Maintains semantic structure

### 5. CodeTextSplitter
```python
from langchain_text_splitters import (
    PythonCodeTextSplitter,
    JavascriptCodeTextSplitter
)

# For Python
splitter = PythonCodeTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# For JavaScript
splitter = JavascriptCodeTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
```
- **Best for:** Source code files
- **Feature:** Respects code structure (functions, classes)

## Alternative Vector Stores

### Currently Used: Chroma

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
```

**Pros:**
- Lightweight and easy to use
- Local persistence
- No external dependencies
- Good for development and small-scale applications

**Cons:**
- Limited scalability
- No distributed capabilities
- Basic querying features

### Alternatives

#### 1. FAISS (Facebook AI Similarity Search)
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

**Best for:**
- High-performance similarity search
- Large-scale applications (millions of vectors)
- GPU acceleration support

**Pros:** Very fast, efficient memory usage
**Cons:** No built-in persistence, requires manual serialization

#### 2. Pinecone (Cloud-based)
```python
from langchain_pinecone import PineconeVectorStore
import pinecone

pinecone.init(
    api_key="your-api-key",
    environment="us-west1-gcp"
)

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="document-index"
)
```

**Best for:**
- Production applications
- Distributed systems
- Scalability requirements

**Pros:** Managed service, highly scalable, real-time updates
**Cons:** Requires API key, costs money, internet dependency

#### 3. Weaviate
```python
from langchain_weaviate import WeaviateVectorStore
import weaviate

client = weaviate.Client(url="http://localhost:8080")

vectorstore = WeaviateVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=client,
    index_name="Document"
)
```

**Best for:**
- Hybrid search (vector + keyword)
- GraphQL queries
- Complex metadata filtering

**Pros:** Feature-rich, supports hybrid search, open-source
**Cons:** Requires separate service, more complex setup

#### 4. Qdrant
```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_db")

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="documents",
    client=client
)
```

**Best for:**
- Production deployments
- Advanced filtering
- Payload-based search

**Pros:** Fast, efficient, good filtering, REST API
**Cons:** Additional service to manage

#### 5. Elasticsearch
```python
from langchain_elasticsearch import ElasticsearchStore

vectorstore = ElasticsearchStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    es_url="http://localhost:9200",
    index_name="document-index"
)
```

**Best for:**
- Existing Elasticsearch infrastructure
- Hybrid search needs
- Full-text search + vector search

**Pros:** Mature ecosystem, powerful querying, analytics
**Cons:** Resource-intensive, complex configuration

## Vector Store Comparison

| Vector Store | Scalability | Setup Complexity | Cost | Best Use Case |
|--------------|-------------|------------------|------|---------------|
| Chroma | Low-Medium | Very Easy | Free | Development, prototyping |
| FAISS | High | Easy | Free | High-performance local |
| Pinecone | Very High | Easy | Paid | Production, cloud |
| Weaviate | High | Medium | Free/Paid | Hybrid search needs |
| Qdrant | High | Medium | Free/Paid | Production deployments |
| Elasticsearch | Very High | Hard | Free/Paid | Existing ES infrastructure |

## Enhanced `ingest()` Function

### With Progress Tracking

```python
from tqdm import tqdm

def ingest():
    print("🔄 Starting document ingestion pipeline...")
    
    # Load documents
    print("\n📂 Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("❌ No documents found.")
        return
    
    print(f"✓ Loaded {len(documents)} documents")
    
    # Split documents
    print("\n✂️  Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✓ Split into {len(chunks)} chunks")
    
    # Generate embeddings and store
    print("\n🔢 Generating embeddings and storing in vector database...")
    embeddings = get_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    # Persist
    print("\n💾 Persisting to disk...")
    vectorstore.persist()
    
    print("\n✅ Ingestion complete!")
    print(f"   Documents: {len(documents)}")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Location: {PERSIST_DIR}")
```

### With Incremental Updates

```python
def ingest(force_rebuild=False):
    """
    Ingest documents with support for incremental updates.
    
    Args:
        force_rebuild: If True, rebuild entire database from scratch
    """
    embeddings = get_embeddings()
    
    # Check if database exists
    db_exists = os.path.exists(PERSIST_DIR)
    
    if force_rebuild and db_exists:
        print("🗑️  Removing existing database...")
        import shutil
        shutil.rmtree(PERSIST_DIR)
        db_exists = False
    
    print("📂 Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("❌ No documents found.")
        return
    
    print(f"✓ Loaded {len(documents)} documents")
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✓ Split into {len(chunks)} chunks")
    
    if db_exists:
        print("📊 Loading existing database...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        
        print("➕ Adding new documents...")
        vectorstore.add_documents(chunks)
    else:
        print("🆕 Creating new database...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
    
    vectorstore.persist()
    print("✅ Chroma DB persisted successfully")
```

### With Batch Processing

```python
def ingest(batch_size=100):
    """
    Ingest documents in batches to manage memory usage.
    
    Args:
        batch_size: Number of chunks to process at once
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
    
    # Process in batches
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=PERSIST_DIR
            )
        else:
            vectorstore.add_documents(batch)
    
    vectorstore.persist()
    print("Chroma DB persisted successfully")
```

## Optimization Strategies

### 1. Parallel Document Loading

```python
from concurrent.futures import ThreadPoolExecutor
import os

def load_single_document(filepath):
    """Load a single document with appropriate loader."""
    filename = os.path.basename(filepath)
    
    try:
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        else:
            return []
        
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = filename
        
        return docs
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def load_documents_parallel(max_workers=4):
    """Load documents in parallel for faster processing."""
    files = [
        os.path.join(DATA_DIR, f) 
        for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f))
    ]
    
    documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(load_single_document, files)
        for docs in results:
            documents.extend(docs)
    
    return documents
```

### 2. Deduplication

```python
def deduplicate_chunks(chunks):
    """Remove duplicate chunks based on content hash."""
    import hashlib
    
    seen_hashes = set()
    unique_chunks = []
    
    for chunk in chunks:
        # Create hash of content
        content_hash = hashlib.md5(
            chunk.page_content.encode()
        ).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
    
    print(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
    return unique_chunks
```

### 3. Content Filtering

```python
def filter_chunks(chunks, min_length=50, max_length=5000):
    """Filter out chunks that are too short or too long."""
    filtered = [
        chunk for chunk in chunks
        if min_length <= len(chunk.page_content) <= max_length
    ]
    
    print(f"Filtered out {len(chunks) - len(filtered)} chunks")
    return filtered
```

## Performance Tuning

### Memory Optimization

```python
# Process documents in smaller batches
BATCH_SIZE = 50

# Reduce chunk overlap
chunk_overlap = 100  # Instead of 200

# Use smaller embedding model
# See embeddings.py documentation
```

### Speed Optimization

```python
# Increase batch size (if memory allows)
BATCH_SIZE = 200

# Use GPU for embeddings
# model_kwargs={'device': 'cuda'}

# Parallel document loading
documents = load_documents_parallel(max_workers=8)

# Use FAISS instead of Chroma for faster similarity search
```

## Common Issues and Solutions

### Issue 1: Out of Memory

**Symptoms:** Process crashes or system becomes unresponsive

**Solutions:**
- Reduce batch size
- Process documents one at a time
- Use smaller chunk size
- Reduce chunk overlap

```python
# Conservative settings
chunk_size = 800
chunk_overlap = 100
batch_size = 25
```

### Issue 2: Slow Processing

**Symptoms:** Ingestion takes too long

**Solutions:**
- Enable GPU for embeddings
- Use parallel document loading
- Use FAISS instead of Chroma
- Reduce number of chunks (increase chunk_size)

### Issue 3: Poor Retrieval Quality

**Symptoms:** Relevant documents not being found

**Solutions:**
- Reduce chunk size (more granular chunks)
- Increase chunk overlap (better context preservation)
- Use better embedding model
- Clean/preprocess documents before ingestion

```python
# Better retrieval settings
chunk_size = 600
chunk_overlap = 150
```

### Issue 4: Encoding Errors

**Symptoms:** UnicodeDecodeError when loading text files

**Solutions:**
```python
# Specify encoding explicitly
loader = TextLoader(path, encoding='utf-8')

# Or try different encodings
try:
    loader = TextLoader(path, encoding='utf-8')
except:
    loader = TextLoader(path, encoding='latin-1')
```

### Issue 5: PDF Extraction Issues

**Symptoms:** Garbled text or missing content from PDFs

**Solutions:**
```python
# Try different PDF loaders
from langchain_community.document_loaders import (
    PyMuPDFLoader,  # Better for complex PDFs
    PDFPlumberLoader,  # Better for tables
    UnstructuredPDFLoader  # Better for scanned PDFs
)

loader = PyMuPDFLoader(path)
# or
loader = PDFPlumberLoader(path)
```

## Best Practices

1. **Always validate documents before ingestion** - Check file sizes, formats, and readability
2. **Use appropriate chunk sizes** - Balance between context and precision
3. **Monitor memory usage** - Especially with large document sets
4. **Add metadata** - Include source, timestamp, file type for better filtering
5. **Implement error handling** - Prevent single file failures from breaking entire pipeline
6. **Test with sample data** - Validate settings before processing large datasets
7. **Version your vector store** - Keep backups before major updates
8. **Log everything** - Track what documents were processed and when
9. **Deduplicate content** - Avoid storing same content multiple times
10. **Incremental updates** - Don't rebuild entire database for small changes

## Usage Examples

### Basic Usage

```bash
python ingest.py
```

### Programmatic Usage

```python
from cortex.ingest import ingest

# Run with defaults
ingest()

# Run with custom settings (if modified)
ingest(force_rebuild=True, batch_size=50)
```

### Integration with Application

```python
# In your main application
from cortex.ingest import ingest

# Initial setup
if not os.path.exists("chroma_db"):
    print("Setting up vector database...")
    ingest()

# Later updates
def update_knowledge_base():
    print("Updating knowledge base...")
    ingest(force_rebuild=False)
```

## Directory Structure

```
project/
├── data/
│   └── documents/          # Place your documents here
│       ├── doc1.pdf
│       ├── doc2.txt
│       └── doc3.docx
├── chroma_db/              # Generated vector store
│   └── [chromadb files]
├── cortex/
│   ├── embeddings.py
│   └── ingest.py
└── main.py
```

## Dependencies

```txt
langchain>=0.1.0
langchain-community>=0.0.1
langchain-text-splitters>=0.0.1
chromadb>=0.4.0
pypdf>=3.0.0
python-docx>=0.8.11
```

Install with:
```bash
pip install langchain langchain-community langchain-text-splitters chromadb pypdf python-docx
```

## Further Reading

- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [RAG Best Practices](https://python.langchain.com/docs/use_cases/question_answering/)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)