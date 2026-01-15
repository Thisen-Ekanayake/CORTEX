from enum import Enum

from cortex.llm import get_llm
from cortex.query import load_qa_chain
from cortex.persona import CORTEX_SYSTEM_PROMPT

class Route(Enum):
    RAG = "rag"
    CHAT = "chat"
    META = "meta"

ROUTER_PROMPT = """
Classify the user query into ONE of the following categories:

- RAG: User is asking for specific information that would be found in documents 
  (e.g., "what does the report say about sales?", "find information about X")
- META: User is asking about the system itself, its capabilities, configuration, or technical details
  (e.g., "what can you do?", "how do you work?", "what are your features?", "tell me about yourself as a system")
- CHAT: General conversation, greetings, reasoning, explanations that don't need documents
  (e.g., "hi", "explain quantum physics", "help me code", "what's the weather like")

Query: "{query}"

Respond with only ONE word: RAG, META, or CHAT.
"""

def route_query(query: str) -> Route:
    """classify query and return appropriate route"""
    llm = get_llm()
    result = llm.invoke(
        ROUTER_PROMPT.format(query=query)
    ).strip().upper()

    if "RAG" in result:
        return Route.RAG
    elif "META" in result:
        return Route.META
    return Route.CHAT   # default to chat for safety

def execute(query: str, callbacks=None):
    """execute query based on routing decision"""
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
        # generate natural response about the system using the system prompt as context
        llm = get_llm(streaming=True, callbacks=callbacks)
        meta_prompt = f"""Based on this system information:
            {CORTEX_SYSTEM_PROMPT}

            Answer the user's question naturally and conversationally.
            Don't just repeat the information verbatim - explain it in a friendly, helpful way.

            User: {query}
            Assistant:"""
        
        if callbacks:
            result_chunks = []
            for chunk in llm.stream(meta_prompt):
                result_chunks.append(chunk)
            return "".join(result_chunks)
        else:
            return llm.invoke(meta_prompt)
    
    # CHAT route
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