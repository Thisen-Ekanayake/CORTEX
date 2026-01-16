from enum import Enum
from typing import Tuple

from cortex.llm import get_llm
from cortex.query import run_rag, run_meta, run_chat


class Route(Enum):
    RAG = "rag"
    CHAT = "chat"
    META = "meta"


ROUTER_PROMPT = """Classify the user query into ONE of the following categories:

- RAG: User is asking for specific information that would be found in documents 
  (e.g., "what does the report say about sales?", "find information about X", "summarize the document")

- META: User is asking about the system itself, its capabilities, configuration, or technical details
  (e.g., "what can you do?", "how do you work?", "what are your features?", "tell me about yourself as a system")

- CHAT: General conversation, greetings, reasoning, explanations that don't need documents
  (e.g., "hi", "explain quantum physics", "help me code", "what do you think about X")

Query: "{query}"

Respond with only ONE word: RAG, META, or CHAT."""


def route_query(query: str) -> Route:
    """Classify query and return appropriate route."""
    llm = get_llm()
    result = llm.invoke(ROUTER_PROMPT.format(query=query)).strip().upper()

    if "RAG" in result:
        return Route.RAG
    elif "META" in result:
        return Route.META
    return Route.CHAT


def execute(query: str, callbacks=None) -> Tuple[str, Route]:
    """
    Execute query based on routing decision.
    
    Args:
        query: User's question/message
        callbacks: Optional callbacks for streaming
    
    Returns:
        tuple: (result_string, route) where route is the Route enum used
    """
    route = route_query(query)

    if route == Route.RAG:
        result = run_rag(query, callbacks=callbacks)
        
        # If no documents found, fall back to CHAT
        if result is None:
            route = Route.CHAT
            result = run_chat(query, callbacks=callbacks)
        
        return result, route
    
    elif route == Route.META:
        result = run_meta(query, callbacks=callbacks)
        return result, route
    
    else:  # Route.CHAT
        result = run_chat(query, callbacks=callbacks)
        return result, route