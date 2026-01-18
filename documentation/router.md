# TF-IDF Query Router Documentation

## Overview

The `router.py` module implements an intelligent query routing system that classifies user queries into three distinct categories and directs them to appropriate handlers. This system uses a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer combined with a machine learning classifier to determine the best route for each query.

## Purpose

This router serves as a critical component in a conversational AI system by:

1. **Analyzing incoming queries** to understand their intent
2. **Classifying queries** into one of three categories (RAG, CHAT, or META)
3. **Routing queries** to the appropriate handler function
4. **Providing confidence scores** to assess routing certainty
5. **Implementing fallback mechanisms** when confidence is low

## Route Categories

The system supports three distinct routes:

### 1. RAG (Retrieval-Augmented Generation)
- **Purpose**: Queries that require retrieving information from a document corpus
- **Use Cases**: 
  - "What does the Q3 report say about sales?"
  - "Find information about our pricing policy"
  - "What are the conclusions in the research paper?"
- **Handler**: `run_rag()` from `cortex.query`

### 2. CHAT
- **Purpose**: General conversational queries that don't require document retrieval
- **Use Cases**:
  - "Hello, how are you?"
  - "Tell me a joke"
  - "What's the weather like?"
- **Handler**: `run_chat()` from `cortex.query`

### 3. META
- **Purpose**: Queries about the system itself, its capabilities, or how to use it
- **Use Cases**:
  - "What can you do?"
  - "How does this system work?"
  - "What are your limitations?"
- **Handler**: `run_meta()` from `cortex.query`

## Architecture

### Class: `TFIDFRouter`

The core routing class that encapsulates the classification logic.

#### Initialization

```python
router = TFIDFRouter(model_dir="tf-idf_classifier/model")
```

**Parameters**:
- `model_dir` (str): Path to directory containing trained model files
  - Default: `"tf-idf_classifier/model"`
  - Must contain:
    - `tfidf.joblib`: Trained TF-IDF vectorizer
    - `classifier.joblib`: Trained classifier model

**Raises**:
- `FileNotFoundError`: If either model file is missing

#### Label Mapping

The router uses a static mapping between numeric labels and Route enums:

```python
LABEL_TO_ROUTE = {
    0: Route.CHAT,
    1: Route.META,
    2: Route.RAG,
}
```

This mapping must align with how the classifier was trained.

### Methods

#### `route_query(query: str)`

Classifies a query and returns the predicted route with confidence scores.

**Parameters**:
- `query` (str): The user's input text

**Returns**:
- `Tuple[Route, Dict[str, float]]`:
  - `Route`: Predicted category (RAG, CHAT, or META)
  - `Dict[str, float]`: Confidence scores for all categories (0-1 range)

**Example**:
```python
route, scores = router.route_query("What can you do?")
# route = Route.META
# scores = {'chat': 0.05, 'meta': 0.92, 'rag': 0.03}
```

## Module-Level Functions

### `get_router(model_dir: str)`

Lazy-loads and returns a global singleton router instance.

**Benefits**:
- Avoids reloading models on every query
- Reduces memory overhead
- Improves response time

**Parameters**:
- `model_dir` (str): Path to model directory

**Returns**:
- `TFIDFRouter`: Global router instance

### `route_query(query: str)`

Convenience function that uses the global router instance.

**Parameters**:
- `query` (str): User's input text

**Returns**:
- `Tuple[Route, Dict[str, float]]`: Route and confidence scores

### `execute(query: str, callbacks=None, confidence_threshold: float = 0.5)`

The main execution function that routes and processes queries.

**Parameters**:
- `query` (str): User's input text
- `callbacks` (optional): Callback functions for streaming responses
- `confidence_threshold` (float): Minimum confidence to trust prediction
  - Default: `0.5` (50%)
  - Range: `0.0` to `1.0`

**Returns**:
- `Tuple[str, Route, Dict[str, float]]`:
  - `str`: Response from the handler
  - `Route`: Route that was actually used
  - `Dict[str, float]`: Confidence scores

**Behavior**:

1. **Routes the query** using the classifier
2. **Checks confidence** against threshold
   - If below threshold → fallback to CHAT
3. **Executes appropriate handler**:
   - RAG: Calls `run_rag()`, falls back to CHAT if no documents found
   - META: Calls `run_meta()`
   - CHAT: Calls `run_chat()`
4. **Returns response** with routing metadata

**Example**:
```python
result, route, scores = execute(
    "What's in the Q3 report?",
    confidence_threshold=0.7
)
```

### `print_routing_info(query: str)`

Debug utility that prints detailed routing information.

**Parameters**:
- `query` (str): User's input text

**Output Example**:
```
Query: "What does the report say about sales?"
Predicted Route: rag
Confidence Scores:
  - rag: 92.62%
  - chat: 5.23%
  - meta: 2.15%
```

## Hyperparameters & Tuning

### 1. Confidence Threshold

**Parameter**: `confidence_threshold` in `execute()`

**Purpose**: Controls how confident the model must be before using predicted route

**Tuning Guidelines**:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.3 - 0.4 | Aggressive routing | High trust in model, diverse queries |
| 0.5 - 0.6 | Balanced (default) | General purpose use |
| 0.7 - 0.8 | Conservative routing | Critical applications, prefer safety |
| 0.9+ | Very conservative | Testing, high-stakes scenarios |

**Impact**:
- **Lower threshold**: More queries routed to predicted category, risk of misrouting
- **Higher threshold**: More fallback to CHAT, safer but less specialized

**Tuning Process**:
1. Analyze confusion matrix on validation set
2. Calculate cost of misrouting each category
3. Set threshold based on acceptable error rate
4. Monitor real-world performance

### 2. TF-IDF Vectorizer Parameters

While these are set during training (not in this file), understanding them helps with model retraining:

**Key Parameters**:
- `max_features`: Number of features to keep (1000-10000)
- `ngram_range`: N-gram range, e.g., `(1, 2)` for unigrams and bigrams
- `min_df`: Minimum document frequency (2-5 for small datasets)
- `max_df`: Maximum document frequency (0.8-0.95 to remove common words)
- `sublinear_tf`: Use sublinear term frequency scaling (recommended: `True`)

### 3. Classifier Parameters

The classifier type isn't specified in the code, but common choices include:

**Logistic Regression** (recommended for TF-IDF):
```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(
    C=1.0,              # Regularization (0.1-10)
    max_iter=1000,      # Iterations for convergence
    class_weight='balanced'  # Handle imbalanced classes
)
```

**Random Forest**:
```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(
    n_estimators=100,    # Number of trees (50-200)
    max_depth=20,        # Tree depth (10-30)
    min_samples_split=5  # Min samples to split (2-10)
)
```

**Support Vector Machine**:
```python
from sklearn.svm import SVC

classifier = SVC(
    kernel='linear',     # Linear kernel works well with TF-IDF
    C=1.0,              # Regularization
    probability=True    # Required for predict_proba()
)
```

## Alternative Approaches

### 1. Semantic Similarity (Sentence Transformers)

**Pros**:
- Understands semantic meaning better than TF-IDF
- Handles paraphrases and synonyms well
- More robust to vocabulary variations

**Cons**:
- Slower inference time
- Requires more computational resources
- Needs GPU for optimal performance

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticRouter:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Store example embeddings for each category
        self.category_embeddings = {
            'rag': [...],
            'chat': [...],
            'meta': [...]
        }
    
    def route_query(self, query):
        query_embedding = self.model.encode(query)
        # Calculate cosine similarity with each category
        # Return highest scoring category
```

### 2. Rule-Based Routing

**Pros**:
- Fast and deterministic
- Easy to understand and debug
- No training required

**Cons**:
- Requires manual rule creation
- Difficult to maintain as system grows
- Poor generalization to unseen queries

**Implementation**:
```python
import re

def rule_based_route(query):
    query_lower = query.lower()
    
    # META rules
    if any(word in query_lower for word in ['can you', 'what can', 'how do you']):
        return Route.META
    
    # RAG rules
    if any(word in query_lower for word in ['report', 'document', 'find', 'search']):
        return Route.RAG
    
    # Default to CHAT
    return Route.CHAT
```

### 3. Large Language Model (LLM) Based Routing

**Pros**:
- Excellent understanding of nuanced queries
- No training data required
- Handles complex, multi-intent queries

**Cons**:
- High latency (100ms - 1s per query)
- Expensive API costs
- Requires internet connection (for cloud models)

**Implementation**:
```python
import anthropic

def llm_route(query):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": f"""Classify this query into one category: RAG, CHAT, or META.
            
            Query: {query}
            
            Respond with only the category name."""
        }]
    )
    return Route[response.content[0].text.strip().upper()]
```

### 4. Hybrid Approach

Combine multiple methods for best results:

```python
def hybrid_route(query):
    # Fast rule-based pre-filtering
    if len(query.split()) < 3:
        return Route.CHAT
    
    # TF-IDF classification
    route, scores = tfidf_router.route_query(query)
    confidence = scores[route.value]
    
    # Use LLM for low-confidence cases
    if confidence < 0.6:
        route = llm_route(query)
    
    return route
```

## Performance Optimization

### 1. Caching

Cache routing decisions for identical queries:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_route_query(query: str):
    return route_query(query)
```

### 2. Batch Processing

Process multiple queries at once:

```python
def route_queries_batch(queries: list[str]) -> list[Tuple[Route, Dict]]:
    router = get_router()
    X = router.vectorizer.transform(queries)
    predictions = router.classifier.predict(X)
    probabilities = router.classifier.predict_proba(X)
    
    results = []
    for pred, probs in zip(predictions, probabilities):
        route = router.LABEL_TO_ROUTE[pred]
        scores = {
            Route.CHAT.value: float(probs[0]),
            Route.META.value: float(probs[1]),
            Route.RAG.value: float(probs[2]),
        }
        results.append((route, scores))
    
    return results
```

### 3. Model Optimization

For production deployments:

- **Quantization**: Reduce model size with `sklearn-onnx`
- **Feature Selection**: Remove low-importance features
- **Model Compression**: Use smaller classifier architectures

## Error Handling & Monitoring

### Recommended Additions

```python
import logging

logger = logging.getLogger(__name__)

def execute_with_monitoring(query: str, **kwargs):
    try:
        result, route, scores = execute(query, **kwargs)
        
        # Log routing decision
        logger.info(f"Query routed to {route.value}", extra={
            'query_length': len(query),
            'confidence': scores[route.value],
            'all_scores': scores
        })
        
        return result, route, scores
        
    except Exception as e:
        logger.error(f"Routing failed: {e}", extra={'query': query})
        # Fallback to CHAT on any error
        return run_chat(query), Route.CHAT, {}
```

### Metrics to Track

1. **Route Distribution**: Percentage of queries per route
2. **Confidence Distribution**: Histogram of confidence scores
3. **Fallback Rate**: How often confidence threshold triggers fallback
4. **Response Time**: Latency for each route type
5. **User Feedback**: Implicit signals (retry rate, follow-up questions)

## Best Practices

### 1. Training Data Quality

- **Balance classes**: Ensure roughly equal examples per category
- **Diverse examples**: Include varied phrasing and vocabulary
- **Edge cases**: Add ambiguous queries with clear labels
- **Regular updates**: Retrain monthly with new real-world queries

### 2. Threshold Selection

- **A/B testing**: Test different thresholds with real users
- **Per-route thresholds**: Consider different thresholds per category
- **Dynamic adjustment**: Adapt threshold based on time of day or user type

### 3. Fallback Strategy

The current implementation has a good fallback hierarchy:
1. Low confidence → CHAT
2. RAG fails (no docs) → CHAT
3. CHAT as safe default

### 4. Model Versioning

```python
class TFIDFRouter:
    def __init__(self, model_dir: str = "tf-idf_classifier/model", version: str = "v1"):
        versioned_dir = os.path.join(model_dir, version)
        # Load from versioned directory
```

## Testing & Validation

### Unit Tests

```python
def test_route_query():
    router = TFIDFRouter()
    
    # Test RAG routing
    route, scores = router.route_query("What's in the quarterly report?")
    assert route == Route.RAG
    assert scores[Route.RAG.value] > 0.5
    
    # Test META routing
    route, scores = router.route_query("What can you do?")
    assert route == Route.META
    
    # Test CHAT routing
    route, scores = router.route_query("Hello there!")
    assert route == Route.CHAT
```

### Integration Tests

```python
def test_execute_with_fallback():
    # Test low confidence fallback
    result, route, scores = execute(
        "ambiguous query xyz",
        confidence_threshold=0.9
    )
    assert route == Route.CHAT  # Should fallback
```

## Migration Guide

### From This System to LLM Routing

1. **Gradual rollout**: Route 10% of traffic to LLM initially
2. **Compare decisions**: Log both TF-IDF and LLM routes
3. **Measure performance**: Track accuracy, latency, cost
4. **Full migration**: Switch when LLM proves superior

### From This System to Semantic Similarity

1. **Generate embeddings**: Create embedding cache for all training queries
2. **Parallel routing**: Run both systems, compare results
3. **Optimize threshold**: Find optimal similarity threshold
4. **Switch**: Replace TF-IDF with semantic model

## Conclusion

This TF-IDF-based router provides a fast, lightweight solution for query classification. It's ideal for:

- **Low-latency requirements** (< 10ms routing time)
- **Resource-constrained environments** (CPU-only servers)
- **Predictable costs** (no API fees)
- **Offline operation** (no internet required)

Consider alternatives when:
- Semantic understanding is critical
- Latency is less important than accuracy
- You have GPU resources available
- Budget allows for LLM API costs

The modular design makes it easy to swap out components or upgrade to more sophisticated approaches as needs evolve.