# Query Module Documentation (`query.py`)

## Overview

This module implements a **Retrieval-Augmented Generation (RAG)** system using LangChain, Chroma vector database, and custom LLM integration. It provides three main execution modes: RAG-based question answering, meta-system queries, and general chat functionality.

## Core Components

### 1. Vector Store & Retrieval

#### `get_retriever()`
Returns a singleton retriever instance connected to the Chroma vector database.

**Configuration:**
- **Database:** Chroma (persistent vector store)
- **Storage Location:** `chroma_db/` directory
- **Default Retrieval:** Top 5 most similar documents (`k=5`)

**How It Works:**
1. Initializes embeddings using `get_embeddings()` from the cortex module
2. Connects to persisted Chroma database at `PERSIST_DIR`
3. Creates a retriever with similarity search
4. Caches the retriever globally for reuse

**Tuning Parameters:**

```python
_retriever = _vectorstore.as_retriever(
    search_kwargs={
        "k": 5,  # Number of documents to retrieve (default: 5)
        # Alternatives:
        # "k": 10,  # More context but slower
        # "k": 3,   # Faster but less context
        # "score_threshold": 0.7,  # Minimum similarity score
        # "fetch_k": 20,  # Fetch more docs before filtering
    }
)
```

**Alternative Retrieval Methods:**

1. **MMR (Maximal Marginal Relevance):** Balances relevance and diversity
```python
_retriever = _vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Fetch 20, return 5 most diverse
        "lambda_mult": 0.5,  # 0=diverse, 1=relevant
    }
)
```

2. **Similarity Score Threshold:** Only return docs above a threshold
```python
_retriever = _vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,
        "k": 5,
    }
)
```

3. **Multi-Query Retrieval:** Generate multiple query variations
```python
from langchain.retrievers.multi_query import MultiQueryRetriever

_retriever = MultiQueryRetriever.from_llm(
    retriever=_vectorstore.as_retriever(),
    llm=get_llm()
)
```

---

### 2. Document Processing

#### `retrieve_docs(query: str)`
Retrieves relevant documents from the vector store.

**Returns:** List of Document objects with metadata (source, page, content)

**Usage:**
```python
docs = retrieve_docs("What is machine learning?")
# Returns top 5 most relevant document chunks
```

#### `format_docs(docs) -> str`
Formats retrieved documents into a readable context string.

**Output Format:**
```
[document_name.pdf, page 5]
Content of the first chunk...

[document_name.pdf, page 7]
Content of the second chunk...
```

**Customization Options:**

```python
def format_docs(docs, include_metadata=True, separator="\n\n") -> str:
    chunks = []
    for i, doc in enumerate(docs, 1):
        if include_metadata:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            # Option 1: Current format
            chunks.append(f"[{source}, page {page}]\n{doc.page_content}")
            
            # Option 2: Numbered format
            # chunks.append(f"Document {i} [{source}, p{page}]:\n{doc.page_content}")
            
            # Option 3: XML-style tags
            # chunks.append(f"<document source='{source}' page='{page}'>\n{doc.page_content}\n</document>")
        else:
            chunks.append(doc.page_content)
    
    return separator.join(chunks)
```

---

### 3. RAG Chain

#### `get_rag_chain(callbacks=None)`
Constructs the RAG processing pipeline using LangChain.

**Pipeline Flow:**
```
User Query → Retrieve Docs → Format Context → LLM Prompt → Generate Answer → Parse Output
```

**Components:**
- **LLM:** Streaming-enabled language model
- **Prompt Template:** Instructs the model to answer only from context
- **Context Retrieval:** Lambda function that retrieves and formats docs
- **Output Parser:** Extracts string output from LLM response

**Current Prompt Template:**
```python
prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the context does not contain the answer, respond with "I don't know".

Context:
{context}

Question:
{question}
"""
)
```

**Alternative Prompt Templates:**

1. **More Instructive:**
```python
"""You are a helpful assistant answering questions based on provided documents.

Instructions:
- Answer ONLY using information from the context below
- If the answer is not in the context, say "I don't have that information"
- Cite specific sources when possible
- Be concise and accurate

Context:
{context}

Question: {question}

Answer:"""
```

2. **With Citation Requirements:**
```python
"""Answer the question using the context below. 
Always cite the source and page number in your answer.

Context:
{context}

Question: {question}

Provide your answer with citations in this format: [source, page X]"""
```

3. **Chain-of-Thought:**
```python
"""Use the context below to answer the question.

Context:
{context}

Question: {question}

Think step by step:
1. What information in the context is relevant?
2. How does it answer the question?
3. What is the final answer?

Answer:"""
```

**Hyperparameter Tuning:**

```python
llm = get_llm(
    streaming=True,
    temperature=0.0,      # 0.0 = deterministic, 0.7 = creative
    max_tokens=500,       # Limit response length
    top_p=0.9,           # Nucleus sampling threshold
    callbacks=callbacks
)
```

---

### 4. Execution Functions

#### `run_rag(query: str, callbacks=None) -> Optional[str]`
Executes the full RAG pipeline.

**Behavior:**
- Returns `None` if no relevant documents found
- Streams response if callbacks provided
- Returns complete response otherwise

**Usage:**
```python
# Standard invocation
answer = run_rag("What is the capital of France?")

# With streaming callbacks
from langchain.callbacks import StreamingStdOutCallbackHandler
answer = run_rag("Explain quantum computing", callbacks=[StreamingStdOutCallbackHandler()])
```

**Optimization Options:**

1. **Add Relevance Filtering:**
```python
def run_rag(query: str, callbacks=None, min_score=0.7) -> Optional[str]:
    docs = retrieve_docs(query)
    
    # Filter by relevance score if available
    filtered_docs = [d for d in docs if d.metadata.get('score', 1.0) >= min_score]
    
    if not filtered_docs:
        return None
    
    # Continue with filtered docs...
```

2. **Add Reranking:**
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def run_rag_with_reranking(query: str) -> Optional[str]:
    base_retriever = get_retriever()
    compressor = LLMChainExtractor.from_llm(get_llm())
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    docs = compression_retriever.get_relevant_documents(query)
    # Continue with compressed docs...
```

#### `get_sources(query: str) -> List[str]`
Retrieves source citations for the query without generating an answer.

**Returns:** List of strings in format `"source_name.pdf, page 5"`

**Usage:**
```python
sources = get_sources("machine learning definition")
# ['ml_textbook.pdf, page 12', 'intro_to_ai.pdf, page 45', ...]
```

**Enhancement Options:**

```python
def get_sources(query: str, include_scores=False) -> List[dict]:
    """Enhanced version returning structured source information"""
    docs = retrieve_docs(query)
    sources = []
    
    for doc in docs:
        source_info = {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "preview": doc.page_content[:100] + "...",
        }
        
        if include_scores and 'score' in doc.metadata:
            source_info["relevance_score"] = doc.metadata['score']
        
        sources.append(source_info)
    
    return sources
```

---

#### `run_meta(query: str, callbacks=None) -> str`
Handles questions about the system itself using the `CORTEX_SYSTEM_PROMPT`.

**Use Cases:**
- "What can you do?"
- "How does this system work?"
- "What are your capabilities?"

**Prompt Strategy:**
- Includes full system prompt as context
- Instructs LLM to explain naturally, not repeat verbatim
- Emphasizes conversational, friendly tone

**Optimization:**

```python
def run_meta(query: str, callbacks=None, temperature=0.7) -> str:
    llm = get_llm(streaming=True, callbacks=callbacks, temperature=temperature)
    
    meta_prompt = f"""You are explaining your capabilities based on this system design:

{CORTEX_SYSTEM_PROMPT}

User question: {query}

Provide a clear, friendly explanation. Be honest about limitations.
Use examples if helpful. Keep it conversational.

Response:"""
    
    # Rest of implementation...
```

---

#### `run_chat(query: str, callbacks=None) -> str`
General conversational mode without document retrieval.

**Use Cases:**
- Greetings and small talk
- General knowledge questions
- Questions outside the document scope

**Behavior:**
- Uses system prompt for identity/context
- No document retrieval
- Streams if callbacks provided

**Enhancement Options:**

1. **Add Conversation History:**
```python
def run_chat(query: str, history: List[dict] = None, callbacks=None) -> str:
    llm = get_llm(streaming=True, callbacks=callbacks)
    
    messages = [{"role": "system", "content": CORTEX_SYSTEM_PROMPT}]
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": query})
    
    return llm.invoke(messages)
```

2. **Add Personality Parameters:**
```python
def run_chat(query: str, tone="professional", verbosity="concise", callbacks=None) -> str:
    tone_instructions = {
        "professional": "Maintain a professional, helpful tone.",
        "casual": "Be friendly and conversational.",
        "technical": "Use precise technical language."
    }
    
    chat_prompt = f"""{CORTEX_SYSTEM_PROMPT}

Style: {tone_instructions.get(tone, '')}
Verbosity: {verbosity}

User: {query}
Assistant:"""
    
    # Rest of implementation...
```

---

## Architecture Patterns

### Singleton Pattern
The module uses global caching (`_vectorstore`, `_retriever`, `_rag_chain`) to avoid reinitializing expensive components.

**Pros:**
- Performance optimization
- Resource efficiency
- Consistent state

**Cons:**
- Global state can cause issues in multi-threaded environments
- Harder to test with different configurations

**Alternative: Dependency Injection**
```python
class RAGSystem:
    def __init__(self, persist_dir="chroma_db", k=5):
        self.embeddings = get_embeddings()
        self.vectorstore = Chroma(persist_directory=persist_dir, 
                                   embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def retrieve_docs(self, query: str):
        return self.retriever.invoke(query)
    
    # ... other methods

# Usage
rag = RAGSystem(k=10)
docs = rag.retrieve_docs("example query")
```

---

## Performance Optimization

### 1. Batch Processing
```python
def run_rag_batch(queries: List[str]) -> List[Optional[str]]:
    """Process multiple queries efficiently"""
    chain = get_rag_chain()
    return chain.batch(queries)
```

### 2. Async Support
```python
async def run_rag_async(query: str) -> Optional[str]:
    """Async version for concurrent requests"""
    docs = retrieve_docs(query)
    if not docs:
        return None
    
    chain = get_rag_chain()
    return await chain.ainvoke(query)
```

### 3. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def retrieve_docs_cached(query: str):
    """Cache frequent queries"""
    retriever = get_retriever()
    return retriever.invoke(query)
```

---

## Error Handling Recommendations

```python
def run_rag_safe(query: str, callbacks=None) -> Tuple[Optional[str], Optional[str]]:
    """
    Safe RAG execution with error handling
    Returns: (answer, error_message)
    """
    try:
        docs = retrieve_docs(query)
        
        if not docs:
            return None, "No relevant documents found"
        
        chain = get_rag_chain(callbacks=callbacks)
        answer = chain.invoke(query)
        
        return answer, None
        
    except Exception as e:
        return None, f"Error during RAG execution: {str(e)}"
```

---

## Testing Recommendations

```python
# Unit test example
def test_retrieve_docs():
    query = "test query"
    docs = retrieve_docs(query)
    assert isinstance(docs, list)
    assert all(hasattr(doc, 'page_content') for doc in docs)
    assert all(hasattr(doc, 'metadata') for doc in docs)

# Integration test example
def test_run_rag():
    answer = run_rag("What is Python?")
    assert answer is None or isinstance(answer, str)
```

---

## Common Issues & Solutions

### Issue 1: Slow Retrieval
**Solution:** Reduce `k` value or use similarity threshold
```python
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
```

### Issue 2: Irrelevant Results
**Solution:** Use MMR or increase embedding quality
```python
_retriever = _vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.7}
)
```

### Issue 3: Context Window Overflow
**Solution:** Limit document content or use compression
```python
def format_docs(docs, max_chars_per_doc=500) -> str:
    chunks = []
    for doc in docs:
        content = doc.page_content[:max_chars_per_doc]
        # ... format as before
```

### Issue 4: Memory Leaks with Streaming
**Solution:** Ensure callbacks are properly cleaned up
```python
def run_rag(query: str, callbacks=None) -> Optional[str]:
    try:
        # ... existing code
    finally:
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, 'cleanup'):
                    callback.cleanup()
```

---

## Future Enhancements

1. **Hybrid Search:** Combine vector search with keyword search
2. **Query Expansion:** Automatically generate query variations
3. **Answer Validation:** Verify answers against multiple sources
4. **Feedback Loop:** Learn from user feedback to improve retrieval
5. **Multi-modal Support:** Handle images, tables, and charts
6. **Conversation Memory:** Maintain context across multiple queries

---

## Dependencies

- **langchain_chroma:** Vector database integration
- **langchain_core:** Core LangChain components (prompts, parsers, runnables)
- **cortex.embeddings:** Custom embedding model provider
- **cortex.llm:** Custom LLM provider
- **cortex.persona:** System prompt and personality configuration

---

## Configuration Summary

| Parameter | Default | Purpose | Tuning Range |
|-----------|---------|---------|--------------|
| `k` | 5 | Number of documents to retrieve | 3-10 |
| `score_threshold` | None | Minimum similarity score | 0.5-0.9 |
| `temperature` | Model default | LLM creativity | 0.0-1.0 |
| `max_tokens` | Model default | Response length limit | 100-2000 |
| `lambda_mult` (MMR) | N/A | Diversity vs relevance | 0.0-1.0 |

---

## Quick Reference

```python
# RAG with document retrieval
answer = run_rag("What is X?")

# Get sources only
sources = get_sources("What is X?")

# Meta questions about the system
response = run_meta("What can you do?")

# General chat
response = run_chat("Hello!")

# Custom retrieval
docs = retrieve_docs("machine learning")
formatted = format_docs(docs)
```