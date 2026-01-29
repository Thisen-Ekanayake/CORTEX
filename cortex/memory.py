from dataclasses import dataclass
from datetime import datetime

@dataclass
class MemoryItem:
    """
    Represents a single conversation memory item.
    
    Attributes:
        query: The user's query/question.
        answer: The assistant's response.
        timestamp: Timestamp when the interaction occurred (HH:MM:SS format).
    """
    query: str
    answer: str
    timestamp: str

class ConversationMemory:
    """
    Manages conversation history with a fixed-size buffer.
    
    Maintains a rolling window of conversation items, automatically
    removing oldest items when the maximum capacity is reached.
    """
    
    def __init__(self, max_items=50):
        """
        Initialize the conversation memory.
        
        Args:
            max_items: Maximum number of conversation items to retain (default: 50).
        """
        self.history = []
        self.max_items = max_items

    def add(self, query, answer):
        """
        Add a new conversation item to memory.
        
        Automatically removes oldest items if max_items is exceeded.
        Timestamp is automatically generated for the current time.
        
        Args:
            query: The user's query/question.
            answer: The assistant's response.
        """
        self.history.append(
            MemoryItem(
                query=query,
                answer=answer,
                timestamp=datetime.now().strftime("%H:%M:%S")
            )
        )
        self.history = self.history[-self.max_items:]

    def all_queries(self):
        """
        Get all queries from conversation history.
        
        Returns:
            list: List of all query strings in chronological order.
        """
        return [item.query for item in self.history]