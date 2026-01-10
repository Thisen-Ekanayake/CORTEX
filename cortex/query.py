from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cortex.embeddings import get_embeddings
from cortex.llm import get_llm
from cortex.persona import CORTEX_SYSTEM_PROMPT


PERSIST_DIR = "chroma_db"

_chain_cache = None     # keep chain in memory
_retriever_cache = None

def load_qa_chain():
    global _chain_cache, _retriever_cache
    if _chain_cache and _retriever_cache:
        return _chain_cache, _retriever_cache
    
    embeddings = get_embeddings()
    llm = get_llm()

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}      # top 5
    )

    template = """You are CORTEX - a local, privacy-first AI assistant.
        Answer the question based only on the following context:
        {context}

        {persona_prompt}

        Question: {question}
        Provide clear, concise answers and include source filenames for reference.
        """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            formatted.append(f"[{source}, page {page}]: {doc.page_content}")
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough(), "persona_prompt": RunnableLambda(lambda x: CORTEX_SYSTEM_PROMPT)}
        | prompt
        | llm
        | StrOutputParser()
    )

    _chain_cache = chain
    _retriever_cache = retriever
    return chain, retriever


def ask(query: str):
    chain, retriever = load_qa_chain()
    answer = chain.invoke(query)
    
    docs = retriever._get_relevant_documents(query, run_manager=None)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:\n")
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        print(f"- {source}, page {page}")
