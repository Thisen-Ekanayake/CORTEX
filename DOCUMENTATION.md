# CORTEX - Complete Function Documentation

## Table of Contents
1. [Overview](#overview)
2. [Module: run.py](#module-runpy)
3. [Module: cortex/embeddings.py](#module-cortexembeddingspy)
4. [Module: cortex/ingest.py](#module-cortexingestpy)
5. [Module: cortex/llm.py](#module-cortexllmpy)
6. [Module: cortex/logo.py](#module-cortexlogopy)
7. [Module: cortex/memory.py](#module-cortexmemorypy)
8. [Module: cortex/persona.py](#module-cortexpersonapy)
9. [Module: cortex/query.py](#module-cortexquerypy)
10. [Module: cortex/router.py](#module-cortexrouterpy)
11. [Module: cortex/streaming.py](#module-cortexstreamingpy)
12. [Module: cortex/ui.py](#module-cortexuipy)

---

## Overview

CORTEX is a local, privacy-first AI assistant that uses Retrieval-Augmented Generation (RAG) to answer questions based on ingested documents. The system consists of multiple modules handling embeddings, document ingestion, LLM interactions, query routing, and user interface.

---

## Module: run.py

**File Path:** `/ml/CORTEX/run.py`

**Description:** Main entry point for the CORTEX application. Provides command-line interface for document ingestion and querying.

### Functions

#### `main()`
**Signature:**
```python
def main()
```

**Description:** Main entry point that parses command-line arguments and executes appropriate actions (ingestion or querying).

**Parameters:** None (uses `argparse` to parse command-line arguments)

**Returns:** None

**Command-Line Arguments:**
- `--ingest`: Flag to ingest documents into the vector database
- `--ask <question>`: String argument to ask a question to CORTEX

**Usage Example:**
```python
# From command line:
python run.py --ingest
python run.py --ask "What is artificial intelligence?"

# Or programmatically:
if __name__ == "__main__":
    main()
```

**Calling Code:**
```python
from cortex.ingest import ingest
from cortex.query import ask

# Ingest documents
if args.ingest:
    ingest()

# Ask a question
if args.ask:
    if not os.path.exists("chroma_db/chroma.sqlite3"):
        print("Vector DB not found. Run with --ingest first")
    else:
        logging.basicConfig(filename="query.log", level=logging.INFO)
        logging.info(f"Query asked: {args.ask}")
        ask(args.ask)
```

**Dependencies:**
- `os`
- `argparse`
- `logging`
- `cortex.ingest.ingest`
- `cortex.query.ask` (Note: This function may need to be implemented)

---

## Module: cortex/embeddings.py

**File Path:** `/ml/CORTEX/cortex/embeddings.py`

**Description:** Manages embedding model initialization and provides singleton access to the embeddings model.

### Functions

#### `get_embeddings()`
**Signature:**
```python
def get_embeddings()
```

**Description:** Returns a singleton instance of the HuggingFace embeddings model. Uses lazy initialization - the model is only loaded on first call.

**Parameters:** None

**Returns:** `HuggingFaceEmbeddings` instance (from `langchain_huggingface`)

**Model Used:** `all-MiniLM-L6-v2` (Sentence Transformers model)

**Usage Example:**
```python
from cortex.embeddings import get_embeddings

embeddings = get_embeddings()
# Use embeddings for vector operations
```

**Calling Code:**
```python
# In cortex/ingest.py
from cortex.embeddings import get_embeddings

embeddings = get_embeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)

# In cortex/query.py
from cortex.embeddings import get_embeddings

embeddings = get_embeddings()
_vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
```

**Dependencies:**
- `langchain_huggingface.HuggingFaceEmbeddings`

**Global Variables:**
- `_embeddings`: Cached embeddings instance (module-level singleton)

---

## Module: cortex/ingest.py

**File Path:** `/ml/CORTEX/cortex/ingest.py`

**Description:** Handles document loading, text splitting, and vector database creation.

### Constants

- `DATA_DIR = "data/documents"`: Directory containing source documents
- `PERSIST_DIR = "chroma_db"`: Directory for persisting the vector database

### Functions

#### `load_documents()`
**Signature:**
```python
def load_documents()
```

**Description:** Loads all supported documents from the `DATA_DIR` directory. Supports PDF, TXT, and DOCX file formats.

**Parameters:** None

**Returns:** `List[Document]` - List of LangChain Document objects with metadata

**Supported File Formats:**
- `.pdf` - Uses `PyPDFLoader`
- `.txt` - Uses `TextLoader`
- `.docx` - Uses `Docx2txtLoader`

**Metadata Added:**
- `source`: Filename of the source document

**Usage Example:**
```python
from cortex.ingest import load_documents

documents = load_documents()
print(f"Loaded {len(documents)} documents")
```

**Calling Code:**
```python
# In ingest() function
documents = load_documents()
```

**Dependencies:**
- `os`
- `langchain_community.document_loaders.PyPDFLoader`
- `langchain_community.document_loaders.TextLoader`
- `langchain_community.document_loaders.Docx2txtLoader`

#### `ingest()`
**Signature:**
```python
def ingest()
```

**Description:** Main ingestion function that loads documents, splits them into chunks, creates embeddings, and persists them to ChromaDB.

**Parameters:** None

**Returns:** None

**Process:**
1. Loads documents using `load_documents()`
2. Splits documents into chunks (size: 1200, overlap: 200)
3. Gets embeddings model
4. Creates ChromaDB vectorstore from documents
5. Persists the vectorstore to disk

**Text Splitter Configuration:**
- `chunk_size=1200`: Maximum characters per chunk
- `chunk_overlap=200`: Overlapping characters between chunks

**Usage Example:**
```python
from cortex.ingest import ingest

# Ingest all documents
ingest()
```

**Calling Code:**
```python
# In run.py
from cortex.ingest import ingest

if args.ingest:
    ingest()

# In cortex/ui.py
from cortex.ingest import ingest

def ingest_documents():
    ingest()
```

**Dependencies:**
- `langchain_text_splitters.RecursiveCharacterTextSplitter`
- `langchain_community.vectorstores.Chroma`
- `cortex.embeddings.get_embeddings`

---

## Module: cortex/llm.py

**File Path:** `/ml/CORTEX/cortex/llm.py`

**Description:** Manages Large Language Model (LLM) initialization with GPU/CPU fallback support.

### Functions

#### `get_llm(streaming=False, callbacks=None, cpu_fallback=True)`
**Signature:**
```python
def get_llm(streaming=False, callbacks=None, cpu_fallback=True)
```

**Description:** Returns an LLM instance with automatic GPU/CPU fallback. Uses LlamaCpp for GPU and transformers pipeline for CPU.

**Parameters:**
- `streaming` (bool, optional): Enable streaming output. Default: `False`
- `callbacks` (list, optional): List of callback handlers for streaming. Default: `None`
- `cpu_fallback` (bool, optional): Enable CPU fallback if GPU unavailable. Default: `True`

**Returns:**
- GPU available: `LlamaCpp` instance
- CPU fallback: `pipeline` instance (text-generation)
- Error: Raises `RuntimeError` if no GPU and `cpu_fallback=False`

**GPU Configuration (LlamaCpp):**
- Model: `models/qwen2.5-1.5b-instruct-q4_0.gguf`
- `n_ctx=32768`: Context window size
- `n_batch=64`: Batch size
- `max_tokens=512`: Maximum tokens to generate
- `temperature=0.2`: Sampling temperature
- `n_threads=8`: Number of threads

**CPU Fallback Configuration:**
- Model: `tinylama-1.1B-chat`
- Uses HuggingFace transformers pipeline

**Usage Example:**
```python
from cortex.llm import get_llm

# Get LLM with streaming
llm = get_llm(streaming=True, callbacks=[my_callback])

# Get LLM without streaming
llm = get_llm()
result = llm.invoke("Hello, world!")
```

**Calling Code:**
```python
# In cortex/query.py
from cortex.llm import get_llm

llm = get_llm(streaming=True, callbacks=callbacks)

# In cortex/router.py
from cortex.llm import get_llm

llm = get_llm()
result = llm.invoke(ROUTER_PROMPT.format(query=query))

llm = get_llm(streaming=True, callbacks=callbacks)
```

**Dependencies:**
- `torch`
- `langchain_community.llms.LlamaCpp`
- `transformers.AutoModelForCausalLM`
- `transformers.AutoTokenizer`
- `transformers.pipeline`

---

## Module: cortex/logo.py

**File Path:** `/ml/CORTEX/cortex/logo.py`

**Description:** Provides ASCII art logo generation with various styling options.

### Constants

- `CORTEX_LOGO_TEXT`: Raw ASCII art logo string

### Functions

#### `get_logo(compact=False, futuristic=True, colored=True)`
**Signature:**
```python
def get_logo(compact=False, futuristic=True, colored=True)
```

**Description:** Returns the CORTEX logo with optional styling.

**Parameters:**
- `compact` (bool, optional): Use compact version (currently unused). Default: `False`
- `futuristic` (bool, optional): Use futuristic style (currently same as regular). Default: `True`
- `colored` (bool, optional): Return as Rich Text object with colors. Default: `True`

**Returns:**
- If `colored=True`: `Text` object (Rich library) with styled colors
- If `colored=False`: Plain string

**Color Scheme:**
- Main CORTEX text: `bold bright_cyan`
- "Think": `bold bright_green`
- "Retrieve": `bold bright_blue`
- "Answer": `bold bright_magenta`
- Other lines: `cyan`

**Usage Example:**
```python
from cortex.logo import get_logo
from rich.console import Console

console = Console()
logo = get_logo(colored=True)
console.print(logo)
```

**Calling Code:**
```python
# In cortex/ui.py (indirectly through print_logo)
from cortex.logo import get_logo

# Note: ui.py uses CORTEX_LOGO_TEXT directly for custom rendering
```

**Dependencies:**
- `rich.text.Text`

#### `get_logo_gradient(compact=False)`
**Signature:**
```python
def get_logo_gradient(compact=False)
```

**Description:** Returns logo with a vertical gradient effect (cyan to blue).

**Parameters:**
- `compact` (bool, optional): Use compact version (currently unused). Default: `False`

**Returns:** `Text` object with gradient styling

**Gradient Colors:** `["bright_cyan", "cyan", "bright_blue", "blue"]`

**Usage Example:**
```python
from cortex.logo import get_logo_gradient
from rich.console import Console

console = Console()
logo = get_logo_gradient()
console.print(logo)
```

**Dependencies:**
- `rich.text.Text`

#### `get_logo_neon(compact=False)`
**Signature:**
```python
def get_logo_neon(compact=False)
```

**Description:** Returns logo with neon-style coloring (bright cyan with bright effects).

**Parameters:**
- `compact` (bool, optional): Use compact version (currently unused). Default: `False`

**Returns:** `Text` object with neon styling

**Color Scheme:**
- Main text: `bold bright_cyan`
- Tagline: `bold bright_green`
- Other lines: `dim cyan`

**Usage Example:**
```python
from cortex.logo import get_logo_neon
from rich.console import Console

console = Console()
logo = get_logo_neon()
console.print(logo)
```

**Dependencies:**
- `rich.text.Text`

---

## Module: cortex/memory.py

**File Path:** `/ml/CORTEX/cortex/memory.py`

**Description:** Manages conversation history and memory storage.

### Classes

#### `MemoryItem`
**Signature:**
```python
@dataclass
class MemoryItem:
    query: str
    answer: str
    timestamp: str
```

**Description:** Data class representing a single conversation item.

**Attributes:**
- `query` (str): User's query
- `answer` (str): Assistant's response
- `timestamp` (str): Timestamp in format "%H:%M:%S"

**Usage Example:**
```python
from cortex.memory import MemoryItem

item = MemoryItem(
    query="What is AI?",
    answer="AI is...",
    timestamp="14:30:00"
)
```

#### `ConversationMemory`
**Signature:**
```python
class ConversationMemory:
    def __init__(self, max_items=50):
```

**Description:** Manages conversation history with automatic size limiting.

**Initialization Parameters:**
- `max_items` (int, optional): Maximum number of items to store. Default: `50`

**Attributes:**
- `history` (list): List of `MemoryItem` objects
- `max_items` (int): Maximum items to store

### Methods

#### `add(query, answer)`
**Signature:**
```python
def add(self, query, answer)
```

**Description:** Adds a new conversation item to memory with automatic timestamp.

**Parameters:**
- `query` (str): User's query
- `answer` (str): Assistant's response

**Returns:** None

**Behavior:** Automatically trims history to `max_items` (keeps most recent items)

**Usage Example:**
```python
from cortex.memory import ConversationMemory

memory = ConversationMemory(max_items=50)
memory.add("What is AI?", "AI is artificial intelligence...")
```

**Calling Code:**
```python
# In cortex/ui.py
from cortex.memory import ConversationMemory

memory = ConversationMemory()
memory.add(query, response_buffer)
```

#### `all_queries()`
**Signature:**
```python
def all_queries(self)
```

**Description:** Returns a list of all queries in history.

**Parameters:** None

**Returns:** `List[str]` - List of query strings

**Usage Example:**
```python
from cortex.memory import ConversationMemory

memory = ConversationMemory()
queries = memory.all_queries()
```

**Dependencies:**
- `dataclasses.dataclass`
- `datetime.datetime`

---

## Module: cortex/persona.py

**File Path:** `/ml/CORTEX/cortex/persona.py`

**Description:** Defines the system prompt and persona for CORTEX.

### Constants

#### `CORTEX_SYSTEM_PROMPT`
**Type:** `str`

**Description:** System prompt that defines CORTEX's role and behavior.

**Content:**
```
You are CORTEX - a local, privacy-first AI assistant.

Your role:
- Act as an intelligent office and knowledge assistant
- Answer clearly, concisely, and professionally
- Use documents ONLY when explicitly relevant

If asked who you are:
- State that you are CORTEX, a local AI assistant designed to help with work and knowledge tasks.

Do NOT claim to be a human, student, employee, or real-world individual.
```

**Usage Example:**
```python
from cortex.persona import CORTEX_SYSTEM_PROMPT

full_prompt = f"{CORTEX_SYSTEM_PROMPT}\n\nUser: {query}\nAssistant:"
```

**Calling Code:**
```python
# In cortex/router.py
from cortex.persona import CORTEX_SYSTEM_PROMPT

full_prompt = f"{CORTEX_SYSTEM_PROMPT}\n\nUser: {query}\nAssistant:"
```

---

## Module: cortex/query.py

**File Path:** `/ml/CORTEX/cortex/query.py`

**Description:** Handles document retrieval, RAG chain creation, and query execution.

### Constants

- `PERSIST_DIR = "chroma_db"`: Directory for the persisted vector database

### Global Variables (Module-level Caches)

- `_vectorstore`: Cached Chroma vectorstore instance
- `_retriever`: Cached retriever instance
- `_rag_chain`: Cached RAG chain instance

### Functions

#### `get_retriever()`
**Signature:**
```python
def get_retriever()
```

**Description:** Returns a singleton retriever instance with caching. Creates Chroma vectorstore and retriever on first call.

**Parameters:** None

**Returns:** Retriever instance (from LangChain Chroma)

**Retriever Configuration:**
- `search_kwargs={"k": 5}`: Retrieves top 5 most relevant documents

**Usage Example:**
```python
from cortex.query import get_retriever

retriever = get_retriever()
docs = retriever.invoke("What is AI?")
```

**Calling Code:**
```python
# In retrieve_docs()
retriever = get_retriever()
return retriever.invoke(query)
```

**Dependencies:**
- `langchain_chroma.Chroma`
- `cortex.embeddings.get_embeddings`

#### `retrieve_docs(query: str)`
**Signature:**
```python
def retrieve_docs(query: str)
```

**Description:** Retrieves relevant documents for a given query.

**Parameters:**
- `query` (str): Search query string

**Returns:** `List[Document]` - List of relevant document chunks

**Usage Example:**
```python
from cortex.query import retrieve_docs

docs = retrieve_docs("What is machine learning?")
```

**Calling Code:**
```python
# In get_rag_chain()
docs = retrieve_docs(query)

# In run_rag()
docs = retrieve_docs(query)

# In get_sources()
docs = retrieve_docs(query)
```

**Dependencies:**
- `get_retriever()`

#### `format_docs(docs) -> str`
**Signature:**
```python
def format_docs(docs) -> str
```

**Description:** Formats a list of documents into a readable string with source metadata.

**Parameters:**
- `docs`: List of Document objects

**Returns:** `str` - Formatted string with document content and metadata

**Format:**
```
[source_filename, page N/A]
content...

[source_filename, page N/A]
content...
```

**Usage Example:**
```python
from cortex.query import format_docs, retrieve_docs

docs = retrieve_docs("What is AI?")
formatted = format_docs(docs)
```

**Calling Code:**
```python
# In get_rag_chain() -> retrieve_and_format()
docs = retrieve_docs(query)
return format_docs(docs)
```

#### `get_rag_chain(callbacks=None)`
**Signature:**
```python
def get_rag_chain(callbacks=None)
```

**Description:** Creates and returns a RAG (Retrieval-Augmented Generation) chain with caching. The chain retrieves documents, formats them, and generates answers using the LLM.

**Parameters:**
- `callbacks` (list, optional): List of callback handlers for streaming. Default: `None`

**Returns:** LangChain Runnable chain

**Chain Structure:**
1. Retrieves and formats documents based on query
2. Passes context and question to prompt template
3. Generates answer using LLM
4. Parses output as string

**Prompt Template:**
```
Answer the question using ONLY the context below.
If the context does not contain the answer, respond with "I don't know".

Context:
{context}

Question:
{question}
```

**Usage Example:**
```python
from cortex.query import get_rag_chain

chain = get_rag_chain(callbacks=[my_callback])
result = chain.invoke("What is AI?")
```

**Calling Code:**
```python
# In run_rag()
chain = get_rag_chain(callbacks=callbacks)
if callbacks:
    for chunk in chain.stream(query):
        chunks.append(chunk)
    return "".join(chunks)
return chain.invoke(query)
```

**Dependencies:**
- `langchain_core.prompts.ChatPromptTemplate`
- `langchain_core.runnables.RunnableLambda`
- `langchain_core.runnables.RunnablePassthrough`
- `langchain_core.output_parsers.StrOutputParser`
- `cortex.llm.get_llm`
- `retrieve_docs`
- `format_docs`

#### `run_rag(query: str, callbacks=None) -> Optional[str]`
**Signature:**
```python
def run_rag(query: str, callbacks=None) -> Optional[str]
```

**Description:** Executes RAG only if relevant documents exist. Returns None if no documents are found.

**Parameters:**
- `query` (str): User's question
- `callbacks` (list, optional): List of callback handlers for streaming. Default: `None`

**Returns:**
- `str`: Generated answer if documents found
- `None`: If no relevant documents found

**Usage Example:**
```python
from cortex.query import run_rag

result = run_rag("What is AI?", callbacks=[my_callback])
if result:
    print(result)
else:
    print("No relevant documents found")
```

**Calling Code:**
```python
# Currently not directly called, but available for use
```

**Dependencies:**
- `retrieve_docs`
- `get_rag_chain`

#### `get_sources(query: str) -> List[str]`
**Signature:**
```python
def get_sources(query: str) -> List[str]
```

**Description:** Retrieves source information for documents matching a query.

**Parameters:**
- `query` (str): Search query

**Returns:** `List[str]` - List of source strings in format "filename, page N/A"

**Usage Example:**
```python
from cortex.query import get_sources

sources = get_sources("What is AI?")
for source in sources:
    print(source)
```

**Calling Code:**
```python
# Available for use but not currently called in codebase
```

**Dependencies:**
- `retrieve_docs`

---

## Module: cortex/router.py

**File Path:** `/ml/CORTEX/cortex/router.py`

**Description:** Routes queries between RAG (document-based) and CHAT (general conversation) modes.

### Classes

#### `Route`
**Signature:**
```python
class Route(Enum):
    RAG = "rag"
    CHAT = "chat"
```

**Description:** Enumeration of routing options.

**Values:**
- `RAG`: Route to document-based retrieval
- `CHAT`: Route to general conversation

### Constants

#### `ROUTER_PROMPT`
**Type:** `str`

**Description:** Prompt template for query classification.

**Content:**
```
Classify the user query into ONE of the following categories:

- RAG: User is asking for specific information that would be found in documents 
  (e.g., "what does the report say about sales?", "find information about X")
- CHAT: General conversation, greetings, questions about the assistant, 
  reasoning, explanations that don't need documents
  (e.g., "hi", "who are you?", "explain quantum physics", "help me code")

Query: "{query}"

Respond with only ONE word: RAG or CHAT.
```

### Functions

#### `route_query(query: str) -> Route`
**Signature:**
```python
def route_query(query: str) -> Route
```

**Description:** Classifies a query and returns the appropriate route (RAG or CHAT).

**Parameters:**
- `query` (str): User's query string

**Returns:** `Route` enum value (`Route.RAG` or `Route.CHAT`)

**Behavior:** Defaults to `Route.CHAT` if classification fails (safety fallback)

**Usage Example:**
```python
from cortex.router import route_query, Route

route = route_query("What is in the documents?")
if route == Route.RAG:
    print("Will use RAG")
else:
    print("Will use CHAT")
```

**Calling Code:**
```python
# In execute()
route = route_query(query)
```

**Dependencies:**
- `cortex.llm.get_llm`

#### `execute(query: str, callbacks=None)`
**Signature:**
```python
def execute(query: str, callbacks=None)
```

**Description:** Executes a query based on routing decision. Routes to RAG or CHAT mode accordingly.

**Parameters:**
- `query` (str): User's query string
- `callbacks` (list, optional): List of callback handlers for streaming. Default: `None`

**Returns:** `str` - Generated response

**Behavior:**
- If `Route.RAG`: Uses RAG chain with document retrieval
- If `Route.CHAT`: Uses LLM directly with system prompt
- Supports streaming when callbacks are provided

**Note:** This function references `load_qa_chain()` which may need to be implemented or should be `get_rag_chain()`.

**Usage Example:**
```python
from cortex.router import execute

# With streaming
def on_token(token):
    print(token, end="", flush=True)

result = execute("What is AI?", callbacks=[StreamHandler(on_token)])

# Without streaming
result = execute("What is AI?")
print(result)
```

**Calling Code:**
```python
# In cortex/ui.py
from cortex.router import execute

result = execute(query, callbacks=[stream_handler])
```

**Dependencies:**
- `cortex.llm.get_llm`
- `cortex.query.load_qa_chain` (Note: May need implementation or should be `get_rag_chain`)
- `cortex.persona.CORTEX_SYSTEM_PROMPT`

---

## Module: cortex/streaming.py

**File Path:** `/ml/CORTEX/cortex/streaming.py`

**Description:** Provides callback handler for streaming LLM tokens in real-time.

### Classes

#### `StreamHandler`
**Signature:**
```python
class StreamHandler(BaseCallbackHandler):
    def __init__(self, on_token):
```

**Description:** Callback handler for streaming LLM tokens in real-time.

**Initialization Parameters:**
- `on_token` (callable): Function to call when a new token is generated. Should accept `token: str` parameter.

**Attributes:**
- `on_token` (callable): Token callback function

### Methods

#### `on_llm_new_token(self, token, **kwargs)`
**Signature:**
```python
def on_llm_new_token(self, token, **kwargs)
```

**Description:** Called automatically by LangChain when a new token is generated by the LLM.

**Parameters:**
- `token` (str): Newly generated token
- `**kwargs`: Additional keyword arguments (ignored)

**Returns:** None

**Usage Example:**
```python
from cortex.streaming import StreamHandler

def on_token(token: str):
    print(token, end="", flush=True)

handler = StreamHandler(on_token)
llm = get_llm(streaming=True, callbacks=[handler])
```

**Calling Code:**
```python
# In cortex/ui.py
from cortex.streaming import StreamHandler

def on_token(token: str):
    token_queue.put(token)

stream_handler = StreamHandler(on_token)
result = execute(query, callbacks=[stream_handler])
```

**Dependencies:**
- `langchain_core.callbacks.BaseCallbackHandler`

---

## Module: cortex/ui.py

**File Path:** `/ml/CORTEX/cortex/ui.py`

**Description:** Provides interactive command-line interface with Rich formatting, streaming responses, and command handling.

### Global Variables

- `console`: Rich Console instance
- `memory`: `ConversationMemory` instance
- `completer`: `WordCompleter` for command completion
- `prompt_style`: Prompt styling configuration

### Constants

- `COMMANDS`: List of available commands: `['help', 'exit', 'quit', 'clear', 'history', 'ingest', 'status']`

### Functions

#### `print_logo(animate=False)`
**Signature:**
```python
def print_logo(animate=False)
```

**Description:** Displays the CORTEX logo with styling in a Rich Panel.

**Parameters:**
- `animate` (bool, optional): Enable animated logo appearance. Default: `False`

**Returns:** None

**Usage Example:**
```python
from cortex.ui import print_logo

print_logo(animate=True)
```

**Calling Code:**
```python
# In interactive_mode()
print_logo(animate=False)

# In process_query() for 'clear' command
print_logo()
```

**Dependencies:**
- `rich.console.Console`
- `rich.panel.Panel`
- `rich.text.Text`
- `cortex.logo.CORTEX_LOGO_TEXT`

#### `print_welcome()`
**Signature:**
```python
def print_welcome()
```

**Description:** Displays welcome message in a Rich Panel.

**Returns:** None

**Usage Example:**
```python
from cortex.ui import print_welcome

print_welcome()
```

**Calling Code:**
```python
# In interactive_mode()
print_welcome()
```

**Dependencies:**
- `rich.console.Console`
- `rich.panel.Panel`
- `rich.text.Text`

#### `print_help()`
**Signature:**
```python
def print_help()
```

**Description:** Displays help information in a Rich Table format.

**Returns:** None

**Usage Example:**
```python
from cortex.ui import print_help

print_help()
```

**Calling Code:**
```python
# In process_query() for 'help' command
print_help()
```

**Dependencies:**
- `rich.console.Console`
- `rich.table.Table`

#### `check_db_status()`
**Signature:**
```python
def check_db_status()
```

**Description:** Checks if the vector database exists.

**Returns:** `bool` - True if database exists, False otherwise

**Usage Example:**
```python
from cortex.ui import check_db_status

if check_db_status():
    print("Database ready")
```

**Calling Code:**
```python
# In show_status()
db_exists = check_db_status()

# In interactive_mode()
if not check_db_status():
    console.print("[yellow]⚠ Warning: Vector database not found.[/yellow]")

# In process_query()
if not check_db_status():
    console.print("[yellow]⚠ Vector database not found...[/yellow]")
```

**Dependencies:**
- `pathlib.Path`

#### `show_status()`
**Signature:**
```python
def show_status()
```

**Description:** Displays system status in a Rich Table format.

**Returns:** None

**Shows:**
- Vector Database status
- Conversation History count
- Memory max items

**Usage Example:**
```python
from cortex.ui import show_status

show_status()
```

**Calling Code:**
```python
# In process_query() for 'status' command
show_status()
```

**Dependencies:**
- `rich.console.Console`
- `rich.table.Table`
- `check_db_status()`
- `memory` (global)

#### `show_history()`
**Signature:**
```python
def show_history()
```

**Description:** Displays conversation history in a Rich Table format (last 10 items).

**Returns:** None

**Usage Example:**
```python
from cortex.ui import show_history

show_history()
```

**Calling Code:**
```python
# In process_query() for 'history' command
show_history()
```

**Dependencies:**
- `rich.console.Console`
- `rich.table.Table`
- `memory` (global)

#### `ingest_documents()`
**Signature:**
```python
def ingest_documents()
```

**Description:** Ingests documents with progress display using Rich Progress.

**Returns:** None

**Usage Example:**
```python
from cortex.ui import ingest_documents

ingest_documents()
```

**Calling Code:**
```python
# In process_query() for 'ingest' command
ingest_documents()
```

**Dependencies:**
- `rich.console.Console`
- `rich.progress.Progress`
- `cortex.ingest.ingest`

#### `stream_response(query: str)`
**Signature:**
```python
def stream_response(query: str)
```

**Description:** Streams response with live updates using Rich Live display. Handles token streaming, displays sources, and saves to memory.

**Parameters:**
- `query` (str): User's query

**Returns:** None

**Process:**
1. Creates token queue and stream handler
2. Executes query in background thread
3. Displays streaming response in real-time
4. Shows document sources after completion
5. Saves to conversation memory

**Usage Example:**
```python
from cortex.ui import stream_response

stream_response("What is AI?")
```

**Calling Code:**
```python
# In process_query()
stream_response(query)
```

**Dependencies:**
- `queue.Queue`
- `threading.Thread`
- `rich.console.Console`
- `rich.live.Live`
- `rich.panel.Panel`
- `rich.markdown.Markdown`
- `rich.spinner.Spinner`
- `cortex.router.execute`
- `cortex.streaming.StreamHandler`
- `cortex.query.load_qa_chain` (Note: May need implementation)
- `memory` (global)

#### `process_query(query: str)`
**Signature:**
```python
def process_query(query: str)
```

**Description:** Processes a user query, handling commands or routing to stream_response.

**Parameters:**
- `query` (str): User's input

**Returns:** None

**Commands Handled:**
- `exit` / `quit`: Exits the application
- `help`: Shows help
- `clear`: Clears screen and shows logo
- `history`: Shows conversation history
- `ingest`: Ingests documents
- `status`: Shows system status
- Other: Routes to `stream_response()`

**Usage Example:**
```python
from cortex.ui import process_query

process_query("help")
process_query("What is AI?")
```

**Calling Code:**
```python
# In interactive_mode()
process_query(query)
```

**Dependencies:**
- `sys`
- `rich.console.Console`
- `check_db_status()`
- `stream_response()`
- Various command handlers

#### `interactive_mode()`
**Signature:**
```python
def interactive_mode()
```

**Description:** Main interactive CLI loop with prompt session, history, and command completion.

**Returns:** None (runs until exit)

**Features:**
- Logo and welcome display
- Database status check
- Logging setup
- Command history (saved to `~/.cortex_history`)
- Auto-suggestions from history
- Command completion
- Custom prompt styling

**Usage Example:**
```python
from cortex.ui import interactive_mode

if __name__ == "__main__":
    interactive_mode()
```

**Calling Code:**
```python
# In cortex/ui.py __main__ block
if __name__ == "__main__":
    interactive_mode()
```

**Dependencies:**
- `logging`
- `pathlib.Path`
- `prompt_toolkit.PromptSession`
- `prompt_toolkit.history.FileHistory`
- `prompt_toolkit.auto_suggest.AutoSuggestFromHistory`
- `prompt_toolkit.completion.WordCompleter`
- `prompt_toolkit.styles.Style`
- `prompt_toolkit.formatted_text.HTML`
- `print_logo()`
- `print_welcome()`
- `check_db_status()`
- `process_query()`

---

## Function Call Graph

### Main Entry Points
```
run.py:main()
  ├─> cortex.ingest:ingest()
  │     ├─> cortex.ingest:load_documents()
  │     ├─> cortex.embeddings:get_embeddings()
  │     └─> Chroma.from_documents()
  │
  └─> cortex.query:ask() [Note: May need implementation]

cortex/ui.py:interactive_mode()
  ├─> print_logo()
  ├─> print_welcome()
  ├─> check_db_status()
  └─> process_query()
        ├─> stream_response()
        │     ├─> cortex.router:execute()
        │     │     ├─> cortex.router:route_query()
        │     │     │     └─> cortex.llm:get_llm()
        │     │     │
        │     │     ├─> cortex.query:load_qa_chain() [Note: May need implementation]
        │     │     │     └─> cortex.query:get_rag_chain()
        │     │     │           ├─> cortex.llm:get_llm()
        │     │     │           ├─> cortex.query:retrieve_docs()
        │     │     │           │     └─> cortex.query:get_retriever()
        │     │     │           │           └─> cortex.embeddings:get_embeddings()
        │     │     │           └─> cortex.query:format_docs()
        │     │     │
        │     │     └─> cortex.llm:get_llm()
        │     │
        │     └─> cortex.streaming:StreamHandler()
        │
        ├─> print_help()
        ├─> show_status()
        ├─> show_history()
        ├─> ingest_documents()
        │     └─> cortex.ingest:ingest()
        └─> print_logo()
```

---

## Notes and Warnings

### Missing Functions
1. **`cortex.query.ask()`**: Referenced in `run.py` but not implemented. May need to be created or replaced with `run_rag()` or `execute()`.

2. **`cortex.query.load_qa_chain()`**: Referenced in `cortex/router.py` and `cortex/ui.py` but not implemented. Should likely be `get_rag_chain()` or needs to return a tuple `(chain, retriever)`.

### Potential Issues
1. **Router.py Line 41**: Calls `load_qa_chain(callbacks=callbacks)` expecting a tuple, but `get_rag_chain()` only returns a chain.

2. **UI.py Line 372**: Calls `load_qa_chain()` expecting a tuple `(chain, retriever)`, but this function doesn't exist.

### Recommendations
1. Implement `ask()` function in `cortex/query.py`:
   ```python
   def ask(query: str):
       result = execute(query)
       print(result)
   ```

2. Either implement `load_qa_chain()` or update references to use `get_rag_chain()` and `get_retriever()` separately.

---

## Dependencies Summary

### External Libraries
- `langchain` - Core LangChain framework
- `langchain-community` - Community integrations
- `langchain-text-splitters` - Text splitting utilities
- `langchain-chroma` - ChromaDB integration
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `pypdf` - PDF processing
- `python-docx` - DOCX processing
- `llama-cpp-python` - LLM inference
- `transformers` - HuggingFace transformers
- `torch` - PyTorch (for GPU detection)
- `rich` - Rich text and beautiful formatting
- `prompt-toolkit` - Interactive CLI
- `textual` - TUI framework (listed but not used)

---

## Version Information

**Documentation Version:** 1.0  
**Last Updated:** Generated from codebase analysis  
**Codebase:** CORTEX v1.0

---

## License

This documentation is generated from the CORTEX codebase. Please refer to the project's license for usage terms.
