# Conversation Memory Documentation

## Overview

The `memory.py` module implements a conversation history management system that stores and manages past interactions between the user and CORTEX. It provides a simple, efficient way to maintain context across multiple exchanges in a conversation.

## Purpose

This module serves as the short-term memory component of the system by:

1. **Storing conversation history** with query-answer pairs
2. **Maintaining temporal context** with timestamps
3. **Managing memory limits** to prevent unbounded growth
4. **Enabling context-aware responses** by providing access to past interactions
5. **Supporting conversation continuity** across multiple turns

## Architecture

### Data Structure: `MemoryItem`

A dataclass representing a single conversation exchange.

```python
@dataclass
class MemoryItem:
    query: str      # User's input/question
    answer: str     # CORTEX's response
    timestamp: str  # Time of interaction (HH:MM:SS format)
```

**Fields**:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `query` | `str` | User's input question or message | "What's in the Q3 report?" |
| `answer` | `str` | CORTEX's complete response | "The Q3 report shows..." |
| `timestamp` | `str` | Time when interaction occurred | "14:23:45" |

**Design Rationale**:
- Uses `@dataclass` decorator for automatic `__init__`, `__repr__`, and comparison methods
- Stores timestamp as string (HH:MM:SS) for easy display, not datetime object
- Simple, flat structure for efficient serialization and retrieval

### Class: `ConversationMemory`

Main class for managing conversation history.

#### Initialization

```python
memory = ConversationMemory(max_items=50)
```

**Parameters**:
- `max_items` (int): Maximum number of conversation items to retain
  - Default: `50`
  - Implements a sliding window to prevent memory overflow
  - Older items are automatically discarded when limit is reached

**Attributes**:
- `history` (list): List of `MemoryItem` objects in chronological order
- `max_items` (int): Maximum capacity of the history

#### Methods

##### `add(query: str, answer: str)`

Adds a new conversation exchange to the memory.

**Parameters**:
- `query` (str): User's input question
- `answer` (str): CORTEX's response

**Behavior**:
1. Creates a new `MemoryItem` with current timestamp
2. Appends to `history` list
3. Trims history to `max_items` length (keeps most recent items)

**Time Complexity**: O(1) for append, O(n) for trimming (when limit exceeded)

**Example**:
```python
memory = ConversationMemory()
memory.add(
    query="What's the weather today?",
    answer="I don't have access to real-time weather data."
)
```

##### `all_queries()`

Retrieves all stored user queries in chronological order.

**Returns**:
- `list[str]`: List of all query strings from history

**Use Cases**:
- Displaying conversation history to user
- Checking if a question was asked before
- Analyzing query patterns
- Building context for next response

**Example**:
```python
queries = memory.all_queries()
print(queries)
# Output: ['Hello', 'What can you do?', 'Tell me about RAG']
```

## Usage Examples

### 1. Basic Conversation Flow

```python
from memory import ConversationMemory

# Initialize memory
memory = ConversationMemory(max_items=10)

# Simulate conversation
exchanges = [
    ("Hello!", "Hi! I'm CORTEX. How can I help you?"),
    ("What can you do?", "I can help with documents, answer questions, and more."),
    ("Great, thanks!", "You're welcome!")
]

for query, answer in exchanges:
    memory.add(query, answer)

# View conversation history
print(f"Total exchanges: {len(memory.history)}")
print(f"All queries: {memory.all_queries()}")
```

### 2. Integration with Chat System

```python
from memory import ConversationMemory
from router import execute

def chat_with_memory(user_query: str, memory: ConversationMemory):
    """Execute query and store in memory"""
    
    # Get response from router
    answer, route, scores = execute(user_query)
    
    # Store in memory
    memory.add(query=user_query, answer=answer)
    
    # Display with context
    print(f"Route used: {route.value}")
    print(f"Answer: {answer}")
    
    return answer

# Usage
memory = ConversationMemory()
chat_with_memory("What's in the report?", memory)
chat_with_memory("Tell me more about the sales figures", memory)
```

### 3. Context-Aware Responses

```python
from memory import ConversationMemory

def build_context_prompt(query: str, memory: ConversationMemory, context_size: int = 3):
    """Build prompt with recent conversation context"""
    
    # Get recent history
    recent_items = memory.history[-context_size:]
    
    # Format context
    context = "Recent conversation:\n"
    for item in recent_items:
        context += f"User: {item.query}\n"
        context += f"Assistant: {item.answer}\n\n"
    
    # Add current query
    context += f"Current query: {query}\n"
    
    return context

# Usage
memory = ConversationMemory()
memory.add("What's the capital of France?", "Paris")
memory.add("What about Italy?", "Rome")

prompt = build_context_prompt("And Spain?", memory)
print(prompt)
# Includes context from previous questions about capitals
```

### 4. Conversation Summary

```python
from memory import ConversationMemory

def get_conversation_summary(memory: ConversationMemory):
    """Generate summary of conversation"""
    
    if not memory.history:
        return "No conversation history"
    
    first_item = memory.history[0]
    last_item = memory.history[-1]
    
    summary = {
        'total_exchanges': len(memory.history),
        'started_at': first_item.timestamp,
        'last_updated': last_item.timestamp,
        'topics': memory.all_queries()
    }
    
    return summary

# Usage
memory = ConversationMemory()
memory.add("Hello", "Hi!")
memory.add("How are you?", "I'm doing well!")

print(get_conversation_summary(memory))
```

### 5. Conversation Export

```python
import json
from memory import ConversationMemory

def export_conversation(memory: ConversationMemory, filepath: str):
    """Export conversation history to JSON"""
    
    data = [
        {
            'query': item.query,
            'answer': item.answer,
            'timestamp': item.timestamp
        }
        for item in memory.history
    ]
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(data)} items to {filepath}")

def import_conversation(filepath: str) -> ConversationMemory:
    """Import conversation history from JSON"""
    
    memory = ConversationMemory()
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for item in data:
        # Reconstruct memory items
        memory.history.append(
            MemoryItem(
                query=item['query'],
                answer=item['answer'],
                timestamp=item['timestamp']
            )
        )
    
    return memory

# Usage
memory = ConversationMemory()
memory.add("Test query", "Test answer")
export_conversation(memory, "conversation.json")

# Later...
loaded_memory = import_conversation("conversation.json")
```

## Advanced Features

### 1. Enhanced Memory with Metadata

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class EnhancedMemoryItem:
    """Extended memory item with additional metadata"""
    query: str
    answer: str
    timestamp: datetime
    route: Optional[str] = None  # Which route was used
    confidence_scores: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'query': self.query,
            'answer': self.answer,
            'timestamp': self.timestamp.isoformat(),
            'route': self.route,
            'confidence_scores': self.confidence_scores,
            'metadata': self.metadata
        }

class EnhancedConversationMemory:
    def __init__(self, max_items=50):
        self.history = []
        self.max_items = max_items
    
    def add(self, query: str, answer: str, route: str = None, 
            confidence_scores: dict = None, **metadata):
        """Add item with extended metadata"""
        self.history.append(
            EnhancedMemoryItem(
                query=query,
                answer=answer,
                timestamp=datetime.now(),
                route=route,
                confidence_scores=confidence_scores,
                metadata=metadata
            )
        )
        self.history = self.history[-self.max_items:]
    
    def get_by_route(self, route: str):
        """Get all interactions that used specific route"""
        return [item for item in self.history if item.route == route]
```

### 2. Memory with Search Capabilities

```python
from memory import ConversationMemory
from typing import List

class SearchableMemory(ConversationMemory):
    """Memory with search functionality"""
    
    def search_queries(self, keyword: str) -> List[MemoryItem]:
        """Search for queries containing keyword"""
        return [
            item for item in self.history 
            if keyword.lower() in item.query.lower()
        ]
    
    def search_answers(self, keyword: str) -> List[MemoryItem]:
        """Search for answers containing keyword"""
        return [
            item for item in self.history 
            if keyword.lower() in item.answer.lower()
        ]
    
    def find_similar_query(self, query: str, threshold: float = 0.7) -> List[MemoryItem]:
        """Find queries similar to given query using simple similarity"""
        from difflib import SequenceMatcher
        
        similar = []
        for item in self.history:
            similarity = SequenceMatcher(None, query.lower(), item.query.lower()).ratio()
            if similarity >= threshold:
                similar.append((item, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in similar]

# Usage
memory = SearchableMemory()
memory.add("What's the weather?", "I don't have weather data")
memory.add("Tell me about the report", "The report shows...")

# Search
results = memory.search_queries("report")
print(f"Found {len(results)} queries about reports")
```

### 3. Time-Based Memory Management

```python
from datetime import datetime, timedelta
from memory import MemoryItem

class TimeBasedMemory:
    """Memory that expires old items based on time"""
    
    def __init__(self, max_age_hours: int = 24):
        self.history = []
        self.max_age = timedelta(hours=max_age_hours)
    
    def add(self, query: str, answer: str):
        """Add item with full datetime"""
        self.history.append(
            MemoryItem(
                query=query,
                answer=answer,
                timestamp=datetime.now().isoformat()  # Store full datetime
            )
        )
        self._cleanup_old_items()
    
    def _cleanup_old_items(self):
        """Remove items older than max_age"""
        cutoff_time = datetime.now() - self.max_age
        
        self.history = [
            item for item in self.history
            if datetime.fromisoformat(item.timestamp) > cutoff_time
        ]
    
    def get_recent(self, hours: int = 1):
        """Get items from last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            item for item in self.history
            if datetime.fromisoformat(item.timestamp) > cutoff
        ]
```

### 4. Persistent Memory with Database

```python
import sqlite3
from memory import MemoryItem
from typing import List

class PersistentMemory:
    """Memory backed by SQLite database"""
    
    def __init__(self, db_path: str = "conversation_memory.db", max_items: int = 50):
        self.db_path = db_path
        self.max_items = max_items
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
    
    def add(self, query: str, answer: str):
        """Add item to database"""
        from datetime import datetime
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        cursor.execute(
            "INSERT INTO conversations (query, answer, timestamp) VALUES (?, ?, ?)",
            (query, answer, timestamp)
        )
        conn.commit()
        
        # Cleanup old items
        cursor.execute(
            f"DELETE FROM conversations WHERE id NOT IN "
            f"(SELECT id FROM conversations ORDER BY id DESC LIMIT {self.max_items})"
        )
        conn.commit()
        conn.close()
    
    def all_queries(self) -> List[str]:
        """Get all queries from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query FROM conversations ORDER BY id")
        queries = [row[0] for row in cursor.fetchall()]
        conn.close()
        return queries
    
    def get_history(self) -> List[MemoryItem]:
        """Get full history as MemoryItem objects"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query, answer, timestamp FROM conversations ORDER BY id")
        
        history = [
            MemoryItem(query=row[0], answer=row[1], timestamp=row[2])
            for row in cursor.fetchall()
        ]
        conn.close()
        return history
```

### 5. Memory with Summarization

```python
from memory import ConversationMemory

class SummarizingMemory(ConversationMemory):
    """Memory that automatically summarizes old conversations"""
    
    def __init__(self, max_items=50, summarize_threshold=20):
        super().__init__(max_items)
        self.summarize_threshold = summarize_threshold
        self.summaries = []
    
    def add(self, query: str, answer: str):
        """Add item and trigger summarization if needed"""
        super().add(query, answer)
        
        if len(self.history) >= self.summarize_threshold:
            self._create_summary()
    
    def _create_summary(self):
        """Create summary of older items and clear them"""
        # Get items to summarize (older half)
        split_point = len(self.history) // 2
        to_summarize = self.history[:split_point]
        
        # Create simple summary
        topics = [item.query for item in to_summarize]
        summary = {
            'timestamp': to_summarize[0].timestamp,
            'topics_covered': topics,
            'num_exchanges': len(to_summarize)
        }
        
        self.summaries.append(summary)
        
        # Keep only recent items
        self.history = self.history[split_point:]
    
    def get_full_context(self):
        """Get both summaries and current history"""
        return {
            'summaries': self.summaries,
            'current_history': self.history
        }
```

## Configuration & Tuning

### 1. Memory Size Optimization

**Choosing `max_items`**:

| Use Case | Recommended Size | Rationale |
|----------|------------------|-----------|
| Quick Q&A | 10-20 | Minimal context needed |
| Document analysis | 30-50 | Moderate context for follow-ups |
| Extended research | 100-200 | Long conversation threads |
| Testing/debugging | 5-10 | Easy to inspect |

**Memory usage estimation**:
```python
import sys

def estimate_memory_usage(memory: ConversationMemory):
    """Estimate memory usage in bytes"""
    total_size = 0
    
    for item in memory.history:
        total_size += sys.getsizeof(item.query)
        total_size += sys.getsizeof(item.answer)
        total_size += sys.getsizeof(item.timestamp)
    
    print(f"Total memory: {total_size / 1024:.2f} KB")
    print(f"Average per item: {total_size / len(memory.history):.2f} bytes")
    
    return total_size
```

### 2. Context Window Configuration

```python
class ConfigurableMemory(ConversationMemory):
    """Memory with configurable context window"""
    
    def __init__(self, max_items=50, context_window=5):
        super().__init__(max_items)
        self.context_window = context_window
    
    def get_recent_context(self, n: int = None):
        """Get recent items for context"""
        window_size = n if n is not None else self.context_window
        return self.history[-window_size:]
    
    def format_context_for_llm(self):
        """Format recent context for LLM prompt"""
        recent = self.get_recent_context()
        
        context = "Previous conversation:\n"
        for item in recent:
            context += f"User: {item.query}\n"
            context += f"Assistant: {item.answer}\n\n"
        
        return context
```

## Alternative Approaches

### 1. Vector-Based Semantic Memory

**Pros**:
- Finds semantically similar past queries
- Better for retrieving relevant context
- Handles paraphrasing

**Cons**:
- Requires embedding model
- Higher computational cost
- More complex implementation

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticMemory:
    def __init__(self, max_items=50):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.history = []
        self.embeddings = []
        self.max_items = max_items
    
    def add(self, query: str, answer: str):
        """Add item and compute embedding"""
        from datetime import datetime
        
        item = MemoryItem(
            query=query,
            answer=answer,
            timestamp=datetime.now().strftime("%H:%M:%S")
        )
        
        # Compute embedding
        embedding = self.model.encode(query)
        
        self.history.append(item)
        self.embeddings.append(embedding)
        
        # Trim if needed
        if len(self.history) > self.max_items:
            self.history = self.history[-self.max_items:]
            self.embeddings = self.embeddings[-self.max_items:]
    
    def find_similar(self, query: str, top_k: int = 3):
        """Find most similar past queries"""
        if not self.history:
            return []
        
        # Encode query
        query_embedding = self.model.encode(query)
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(self.history[i], similarities[i]) for i in top_indices]
```

### 2. Redis-Based Distributed Memory

**Pros**:
- Shared across multiple instances
- Fast access
- Built-in expiration

**Cons**:
- Requires Redis server
- Network overhead
- More complex setup

**Implementation**:
```python
import redis
import json
from datetime import datetime

class RedisMemory:
    def __init__(self, host='localhost', port=6379, session_id='default'):
        self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
        self.session_id = session_id
        self.key_prefix = f"conversation:{session_id}"
    
    def add(self, query: str, answer: str):
        """Add to Redis list"""
        item = {
            'query': query,
            'answer': answer,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        # Store as JSON
        self.redis_client.lpush(self.key_prefix, json.dumps(item))
        
        # Trim to max items
        self.redis_client.ltrim(self.key_prefix, 0, 49)  # Keep 50 items
    
    def all_queries(self):
        """Get all queries from Redis"""
        items_json = self.redis_client.lrange(self.key_prefix, 0, -1)
        items = [json.loads(item) for item in items_json]
        return [item['query'] for item in items]
```

### 3. Graph-Based Conversation Memory

**Pros**:
- Tracks relationships between topics
- Enables topic-based retrieval
- Good for complex conversations

**Cons**:
- Complex to implement
- Higher storage overhead
- Requires graph database or library

**Implementation**:
```python
import networkx as nx

class GraphMemory:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.conversation_count = 0
    
    def add(self, query: str, answer: str, topics: list = None):
        """Add conversation as graph nodes"""
        self.conversation_count += 1
        node_id = f"conv_{self.conversation_count}"
        
        # Add conversation node
        self.graph.add_node(
            node_id,
            query=query,
            answer=answer,
            timestamp=datetime.now().strftime("%H:%M:%S")
        )
        
        # Link to previous conversation
        if self.conversation_count > 1:
            prev_id = f"conv_{self.conversation_count - 1}"
            self.graph.add_edge(prev_id, node_id, relation='follows')
        
        # Add topic nodes if provided
        if topics:
            for topic in topics:
                self.graph.add_node(topic, type='topic')
                self.graph.add_edge(node_id, topic, relation='about')
    
    def get_conversations_about(self, topic: str):
        """Get all conversations about a topic"""
        if topic not in self.graph:
            return []
        
        # Find all conversation nodes connected to this topic
        conversations = []
        for node in self.graph.predecessors(topic):
            if node.startswith('conv_'):
                conv_data = self.graph.nodes[node]
                conversations.append(conv_data)
        
        return conversations
```

## Testing

### 1. Unit Tests

```python
import unittest
from memory import ConversationMemory, MemoryItem

class TestConversationMemory(unittest.TestCase):
    def test_init(self):
        """Test memory initialization"""
        memory = ConversationMemory(max_items=10)
        self.assertEqual(len(memory.history), 0)
        self.assertEqual(memory.max_items, 10)
    
    def test_add_single_item(self):
        """Test adding single item"""
        memory = ConversationMemory()
        memory.add("Hello", "Hi there!")
        
        self.assertEqual(len(memory.history), 1)
        self.assertEqual(memory.history[0].query, "Hello")
        self.assertEqual(memory.history[0].answer, "Hi there!")
    
    def test_max_items_limit(self):
        """Test that max_items limit is enforced"""
        memory = ConversationMemory(max_items=3)
        
        for i in range(5):
            memory.add(f"Query {i}", f"Answer {i}")
        
        # Should only keep last 3
        self.assertEqual(len(memory.history), 3)
        self.assertEqual(memory.history[0].query, "Query 2")
        self.assertEqual(memory.history[-1].query, "Query 4")
    
    def test_all_queries(self):
        """Test all_queries method"""
        memory = ConversationMemory()
        queries = ["Q1", "Q2", "Q3"]
        
        for q in queries:
            memory.add(q, f"Answer to {q}")
        
        self.assertEqual(memory.all_queries(), queries)
    
    def test_timestamp_format(self):
        """Test timestamp is in correct format"""
        memory = ConversationMemory()
        memory.add("Test", "Response")
        
        timestamp = memory.history[0].timestamp
        # Check format HH:MM:SS
        self.assertRegex(timestamp, r'^\d{2}:\d{2}:\d{2}$')
```

### 2. Integration Tests

```python
def test_memory_with_router():
    """Test memory integration with router"""
    from router import execute
    from memory import ConversationMemory
    
    memory = ConversationMemory()
    
    # Simulate conversation
    queries = [
        "What can you do?",
        "Tell me about RAG",
        "How does it work?"
    ]
    
    for query in queries:
        answer, route, scores = execute(query)
        memory.add(query, answer)
    
    # Verify all stored
    assert len(memory.history) == 3
    assert memory.all_queries() == queries
```

### 3. Performance Tests

```python
import time

def test_memory_performance():
    """Test memory performance with large dataset"""
    memory = ConversationMemory(max_items=1000)
    
    # Add 10,000 items
    start_time = time.time()
    for i in range(10000):
        memory.add(f"Query {i}", f"Answer {i}")
    add_time = time.time() - start_time
    
    # Test retrieval
    start_time = time.time()
    queries = memory.all_queries()
    retrieve_time = time.time() - start_time
    
    print(f"Add time: {add_time:.4f}s")
    print(f"Retrieve time: {retrieve_time:.4f}s")
    print(f"Final size: {len(memory.history)}")
    
    assert len(memory.history) == 1000  # Should be trimmed
```

## Best Practices

### 1. Memory Management

✅ **DO**:
- Set appropriate `max_items` based on use case
- Clear memory between different conversation sessions
- Export important conversations before clearing
- Monitor memory usage in long-running applications

❌ **DON'T**:
- Use unbounded memory (always set `max_items`)
- Store sensitive information without encryption
- Keep memory across different users/sessions
- Ignore memory cleanup in long-running processes

### 2. Privacy & Security

```python
class SecureMemory(ConversationMemory):
    """Memory with optional encryption"""
    
    def __init__(self, max_items=50, encrypt=False, key=None):
        super().__init__(max_items)
        self.encrypt = encrypt
        self.key = key
        
        if encrypt and not key:
            raise ValueError("Encryption key required")
    
    def add(self, query: str, answer: str):
        """Add with optional encryption"""
        if self.encrypt:
            # Simple example - use proper encryption in production
            from cryptography.fernet import Fernet
            f = Fernet(self.key)
            query = f.encrypt(query.encode()).decode()
            answer = f.encrypt(answer.encode()).decode()
        
        super().add(query, answer)
```

### 3. Session Management

```python
class SessionMemory:
    """Manage multiple conversation sessions"""
    
    def __init__(self):
        self.sessions = {}
    
    def get_or_create(self, session_id: str) -> ConversationMemory:
        """Get existing or create new session memory"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory()
        return self.sessions[session_id]
    
    def clear_session(self, session_id: str):
        """Clear specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def clear_all(self):
        """Clear all sessions"""
        self.sessions.clear()

# Usage
sessions = SessionMemory()

# User A's conversation
user_a_memory = sessions.get_or_create("user_a")
user_a_memory.add("Hello", "Hi!")

# User B's conversation (separate)
user_b_memory = sessions.get_or_create("user_b")
user_b_memory.add("Help me", "Sure!")
```

## Conclusion

The `memory.py` module provides a simple, effective foundation for conversation management. It's ideal for:

- **Short-term context** in conversational AI
- **Session-based applications** with limited history
- **Lightweight deployments** without database requirements
- **Prototyping and testing** conversation systems

### When to Use vs. Alternatives

| Scenario | Recommended Approach |
|----------|---------------------|
| Simple chatbot | Current `ConversationMemory` |
| Semantic search needed | Vector-based memory |
| Multi-user system | Redis or database-backed |
| Long-term storage | Persistent database memory |
| Complex relationships | Graph-based memory |

The module's simplicity makes it easy to understand, extend, and integrate into larger systems while providing a solid foundation for more sophisticated memory implementations.