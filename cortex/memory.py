from dataclasses import dataclass
from datetime import datetime

@dataclass
class MemoryItem:
    query: str
    answer: str
    timestamp: str

class ConversationMemory:
    def __init__(self, max_items=50):
        self.history = []
        self.max_items = max_items

    def add(self, query, answer):
        self.history.append(
            MemoryItem(
                query=query,
                answer=answer,
                timestamp=datetime.now().strftime("%H:%M:%S")
            )
        )
        self.history = self.history[-self.max_items:]

    def all_queries(self):
        return [item.query for item in self.history]