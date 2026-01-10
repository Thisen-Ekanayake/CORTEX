from enum import Enum

from cortex.llm import get_llm
from cortex.query import load_qa_chain

class Route(Enum):
    RAG = "rag"
    CHAT = "chat"
    META = "meta"

ROUTER_PROMPT = """
Classify the user query into ONE of the following catgeories:

- RAG: requires searching the document knowledge base
- CHAT: general reasoning or explanation, no documents needed
- META: about the system itself

Query: "{query}"

Respond with only one word: RAG, CHAT, or META.
"""

def route_query(query: str) -> Route:
    llm = get_llm()
    result = llm.invoke(
        ROUTER_PROMPT.format(query=query)
    ).strip().upper()

    if "RAG" in result:
        return Route.RAG
    if "META" in result:
        return Route.META
    return Route.CHAT

def execute(query: str):
    route = route_query(query)

    if route == Route.RAG:
        chain, _ = load_qa_chain()
        return chain.invoke(query)
    
    if route == Route.META:
        return (
            "I am CORTEXT - a local knowledge interface built on your documents."
            "I retrieve context-aware answers using a vector database and a local LLM."
        )
    
    llm = get_llm()
    return llm.invoke(query)