from enum import Enum

from cortex.llm import get_llm
from cortex.query import load_qa_chain
from cortex.persona import CORTEX_SYSTEM_PROMPT

class Route(Enum):
    RAG = "rag"
    CHAT = "chat"

ROUTER_PROMPT = """
Classify the user query into ONE of the following categories:

- RAG: User is asking for specific information that would be found in documents 
  (e.g., "what does the report say about sales?", "find information about X")
- CHAT: General conversation, greetings, questions about the assistant, 
  reasoning, explanations that don't need documents
  (e.g., "hi", "who are you?", "explain quantum physics", "help me code")

Query: "{query}"

Respond with only ONE word: RAG or CHAT.
"""

def route_query(query: str) -> Route:
    llm = get_llm()
    result = llm.invoke(
        ROUTER_PROMPT.format(query=query)
    ).strip().upper()

    if "RAG" in result:
        return Route.RAG
    return Route.CHAT   # default to chat for safety

def execute(query: str, callbacks=None):
    route = route_query(query)

    if route == Route.RAG:
        chain, _ = load_qa_chain(callbacks=callbacks)
        if callbacks:
            # use streaming when callbacks are provided
            # callbacks are already attached to the llm in the chain
            result_chunks = []
            for chunk in chain.stream(query):
                result_chunks.append(chunk)
            return "".join(result_chunks)
        else:
            return chain.invoke(query)
    
    if route == Route.META:
        return CORTEX_SYSTEM_PROMPT
    
    llm = get_llm(streaming=True, callbacks=callbacks)
    full_prompt = f"{CORTEX_SYSTEM_PROMPT}\n\nUser: {query}\nAssistant:"
    
    if callbacks:
        # use streaming when callbacks are provided
        # callbacks are already attached to the llm
        result_chunks = []
        for chunk in llm.stream(full_prompt):
            result_chunks.append(chunk)
        return "".join(result_chunks)
    else:
        return llm.invoke(full_prompt)