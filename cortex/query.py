from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cortex.embeddings import get_embeddings
from cortex.llm import get_llm


PERSIST_DIR = "chroma_db"


def load_qa_chain():
    embeddings = get_embeddings()
    llm = get_llm()

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    template = """Answer the question based only on the following context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(query: str):
    chain, retriever = load_qa_chain()
    answer = chain.invoke(query)
    # Prefer public API when available; fall back to private method if necessary
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query)
    else:
        try:
            docs = retriever._get_relevant_documents(query, run_manager=None)
        except TypeError:
            # Older signature may not accept run_manager kw; try positional
            docs = retriever._get_relevant_documents(query)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:\n")
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        print(f"- {source}, page {page}")
