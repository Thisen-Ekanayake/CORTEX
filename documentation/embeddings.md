# embeddings.py Documentation

## Overview

This module provides a singleton pattern implementation for managing text embeddings using HuggingFace's sentence transformers. It uses the `all-MiniLM-L6-v2` model to convert text into dense vector representations, which are commonly used for semantic search, similarity matching, and retrieval-augmented generation (RAG) applications.

## Core Functionality

### `get_embeddings()`

Returns a singleton instance of HuggingFace embeddings model.

**Purpose:** Ensures only one instance of the embeddings model is created and reused throughout the application, saving memory and initialization time.

**Returns:** `HuggingFaceEmbeddings` object configured with the `all-MiniLM-L6-v2` model.

**Usage Example:**
```python
from embeddings import get_embeddings

embeddings = get_embeddings()
vector = embeddings.embed_query("Hello, world!")
vectors = embeddings.embed_documents(["doc1", "doc2", "doc3"])
```

## Current Model: all-MiniLM-L6-v2

### Specifications
- **Dimension:** 384
- **Max Sequence Length:** 256 tokens
- **Model Size:** ~80MB
- **Performance:** Fast inference, suitable for production
- **Quality:** Good balance between speed and accuracy

### Strengths
- Lightweight and fast
- Low memory footprint
- Good for general-purpose embeddings
- Well-suited for real-time applications

### Limitations
- Lower accuracy compared to larger models
- 256 token limit may truncate longer documents
- Not specialized for domain-specific tasks

## Alternative Embedding Models

### 1. **Larger Sentence Transformers**

#### all-mpnet-base-v2
```python
_embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"
)
```
- **Dimension:** 768
- **Max Tokens:** 384
- **Best for:** Higher accuracy requirements
- **Trade-off:** Slower inference, larger memory

#### all-MiniLM-L12-v2
```python
_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L12-v2"
)
```
- **Dimension:** 384
- **Max Tokens:** 256
- **Best for:** Better accuracy than L6 with similar speed
- **Trade-off:** Slightly larger model size

### 2. **Multilingual Models**

#### paraphrase-multilingual-MiniLM-L12-v2
```python
_embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
```
- **Best for:** Multi-language support (50+ languages)
- **Use case:** International applications

### 3. **Domain-Specific Models**

#### BAAI/bge-small-en-v1.5
```python
_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
```
- **Best for:** Retrieval tasks and RAG systems
- **Performance:** Strong performance on MTEB benchmark

#### thenlper/gte-small
```python
_embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-small"
)
```
- **Best for:** General text embeddings with good retrieval performance

### 4. **OpenAI Embeddings** (API-based)
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # or text-embedding-3-large
)
```
- **Best for:** Highest quality embeddings
- **Trade-off:** Requires API key, costs money, network dependency

### 5. **Cohere Embeddings** (API-based)
```python
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(
    model="embed-english-v3.0"
)
```
- **Best for:** Strong retrieval performance
- **Trade-off:** Requires API key and internet connection

## Configuration Options

### Basic Configuration with Hyperparameters

```python
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # or 'cuda' for GPU
    encode_kwargs={'normalize_embeddings': True}  # L2 normalization
)
```

### Available Parameters

#### `model_name` (str)
- The HuggingFace model identifier
- Default: `"sentence-transformers/all-MiniLM-L6-v2"`

#### `cache_folder` (str, optional)
```python
model_kwargs={'cache_folder': '/path/to/cache'}
```
- Specify where to cache downloaded models
- Useful for shared environments or custom paths

#### `model_kwargs` (dict)
```python
model_kwargs={
    'device': 'cuda',  # 'cpu', 'cuda', 'mps' (for Mac)
    'trust_remote_code': False  # Set True for custom models
}
```

#### `encode_kwargs` (dict)
```python
encode_kwargs={
    'normalize_embeddings': True,  # L2 normalize vectors
    'batch_size': 32,  # Batch size for encoding
    'show_progress_bar': False,  # Show progress for large batches
    'convert_to_numpy': True,  # Return numpy arrays
    'convert_to_tensor': False  # Return torch tensors instead
}
```

### Advanced Configuration Example

```python
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cuda',  # Use GPU if available
                'cache_folder': './models_cache'
            },
            encode_kwargs={
                'normalize_embeddings': True,  # Better for cosine similarity
                'batch_size': 32,  # Process 32 texts at once
                'show_progress_bar': True  # Show encoding progress
            }
        )
    return _embeddings
```

## Performance Tuning

### 1. **GPU Acceleration**
```python
model_kwargs={'device': 'cuda'}  # Requires CUDA-capable GPU
```
- Speeds up embedding generation significantly
- Essential for large-scale processing

### 2. **Batch Size Optimization**
```python
encode_kwargs={'batch_size': 64}  # Increase for more throughput
```
- Larger batches = faster processing but more memory
- Tune based on available RAM/VRAM
- Typical values: 16-128

### 3. **Normalization**
```python
encode_kwargs={'normalize_embeddings': True}
```
- Enables cosine similarity instead of dot product
- Generally recommended for semantic search
- Makes vectors unit length (L2 norm = 1)

### 4. **Multi-threading for CPU**
```python
import torch
torch.set_num_threads(4)  # Set before loading model
```

## Use Cases

### 1. **Semantic Search**
```python
embeddings = get_embeddings()
query_vector = embeddings.embed_query("What is machine learning?")
doc_vectors = embeddings.embed_documents(document_list)
# Compare using cosine similarity
```

### 2. **Vector Database Integration**
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=get_embeddings()
)
```

### 3. **Document Clustering**
```python
embeddings = get_embeddings()
vectors = embeddings.embed_documents(documents)
# Use with KMeans, DBSCAN, etc.
```

## Model Selection Guide

| Use Case | Recommended Model | Dimension | Speed |
|----------|------------------|-----------|-------|
| Fast prototyping | all-MiniLM-L6-v2 | 384 | Fast |
| Production RAG | BAAI/bge-small-en-v1.5 | 384 | Fast |
| High accuracy | all-mpnet-base-v2 | 768 | Medium |
| Multilingual | paraphrase-multilingual-MiniLM-L12-v2 | 384 | Medium |
| Best quality | text-embedding-3-large (OpenAI) | 3072 | API |
| Budget-friendly | all-MiniLM-L6-v2 | 384 | Fast |

## Common Issues and Solutions

### Issue 1: Out of Memory
**Solution:** Reduce batch size or use a smaller model
```python
encode_kwargs={'batch_size': 8}  # Reduce from default
```

### Issue 2: Slow Inference
**Solutions:**
- Use GPU: `model_kwargs={'device': 'cuda'}`
- Use smaller model: Switch to `all-MiniLM-L6-v2`
- Increase batch size: `encode_kwargs={'batch_size': 64}`

### Issue 3: Poor Search Quality
**Solutions:**
- Use larger model: `all-mpnet-base-v2`
- Enable normalization: `encode_kwargs={'normalize_embeddings': True}`
- Use domain-specific model for specialized content

### Issue 4: Token Limit Exceeded
**Solution:** Chunk documents before embedding
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Below 256 token limit
    chunk_overlap=20
)
chunks = splitter.split_documents(documents)
```

## Best Practices

1. **Always use the singleton pattern** to avoid loading the model multiple times
2. **Enable normalization** for cosine similarity-based searches
3. **Use GPU** when processing large document collections
4. **Choose model based on requirements:** balance between speed, accuracy, and cost
5. **Cache embeddings** when possible to avoid recomputation
6. **Monitor memory usage** especially with large models or batch sizes
7. **Benchmark different models** with your specific data before production

## Dependencies

```txt
langchain-huggingface>=0.0.1
sentence-transformers>=2.0.0
torch>=1.9.0
```

Install with:
```bash
pip install langchain-huggingface sentence-transformers torch
```

## Further Reading

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [HuggingFace Models Hub](https://huggingface.co/models?pipeline_tag=sentence-similarity)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Compare model performance
- [LangChain Embeddings Guide](https://python.langchain.com/docs/modules/data_connection/text_embedding/)