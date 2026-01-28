# CORTEX

A local, privacy-first RAG (Retrieval-Augmented Generation) system with intelligent query routing and reinforcement learning capabilities.

## Overview

CORTEX is an AI assistant that combines document retrieval with language models to answer questions from your knowledge base. It features intelligent query routing to determine when to use document retrieval (RAG), handle system queries (META), or engage in general conversation (CHAT).

## Core Features

- **Document Ingestion**: Load PDF, TXT, and DOCX files into a vector database (Chroma)
- **Intelligent Query Routing**: TF-IDF classifier automatically routes queries to RAG, CHAT, or META handlers
- **Reinforcement Learning**: Learn from user feedback to improve routing accuracy over time
- **Streaming Responses**: Real-time token streaming for responsive user experience
- **Conversation Memory**: Maintains context of recent interactions
- **Multiple Interfaces**: Rich interactive CLI, minimal CLI, and RL training interface
- **Multimodal Tools**: Image search, speech-to-text, text-to-speech, and vision-language reasoning utilities

## Architecture

### Core Components

- **`embeddings.py`**: HuggingFace embeddings (all-MiniLM-L6-v2) for document vectorization
- **`ingest.py`**: Document loading, text splitting, and Chroma vector store creation
- **`llm.py`**: LLM setup with GPU/CPU fallback (Qwen2.5-1.5B or tinylama)
- **`query.py`**: RAG chain implementation with document retrieval and three execution modes:
  - `run_rag()`: Retrieves relevant documents and generates answers
  - `run_meta()`: Answers questions about the system itself
  - `run_chat()`: General conversation without document retrieval
- **`router.py`**: TF-IDF-based query classifier that routes queries to appropriate handlers
- **`memory.py`**: Conversation history management
- **`persona.py`**: System prompt defining CORTEX's identity and behavior
- **`streaming.py`**: Token streaming handler for real-time output
 
### User Interfaces

- **`ui.py`**: Rich interactive CLI with logo, commands, and streaming display
- **`cli.py`**: Minimal CLI with confidence score visualization
- **`rl_cli.py`**: Interactive RL training interface for improving routing
- **`rl_dashboard.py`**: Visualization dashboard for RL learning progress

### Reinforcement Learning

- **`rl_router.py`**: RL router that learns from user feedback to adjust confidence weights
- Uses Q-learning style updates to improve routing accuracy
- Persists feedback and metrics for continuous learning

### Extended Tools

- **Image Search (`image_search/`)**
  - `realtime_retrieval.py`: Hybrid Google + CLIP pipeline that fetches web images, caches them locally, and re-ranks results with CLIP for semantically relevant matches.
  - `clip_retrieval_local.py`: Index and retrieve images from a local folder using CLIP embeddings.
  - `google_search.py`: Fetch raw image URLs from Google Images for a given text query.
- **Speech-to-Text (`speech_to_text/`)**
  - `record_and_transcribe.py`: One-shot recording and transcription pipeline using a local Parakeet ASR `.nemo` model (GPU recommended).
  - `audio_recorder.py`: Utility for recording microphone audio to WAV.
- **Text-to-Speech (`text_to_speech/`)**
  - `text_to_speech.py`: `TextToSpeech` wrapper around Coqui TTS with file synthesis and local playback helpers.
  - `testing.py`: Minimal example that synthesizes a sample sentence to `logs/output.wav`.
- **Vision-Language Inference (`vl_inference.py`)**
  - Qwen2.5-VL-3B-based script for image understanding and image+text question answering with 4-bit quantization for efficient GPU usage.

## Usage

### Basic Usage

```bash
# Ingest documents into vector database
python run.py --ingest

# Ask a question
python run.py --ask "What is machine learning?"

# Interactive mode (Rich UI)
python -m cortex.ui

# Minimal CLI
python -m cortex.cli

# RL Training mode
python -m cortex.rl_cli

# View RL dashboard
python -m cortex.rl_dashboard
 
# Hybrid image search (Google + CLIP)
python image_search/realtime_retrieval.py --prompt "a golden retriever playing in a park"

# Record audio and transcribe to text (speech-to-text)
python speech_to_text/record_and_transcribe.py

# Simple text-to-speech test (writes WAV file)
python text_to_speech/testing.py
```

### Query Routing

The system automatically routes queries into three categories:

1. **RAG**: Questions requiring document retrieval (e.g., "What does the report say about sales?")
2. **META**: Questions about the system (e.g., "What can you do?", "Who are you?")
3. **CHAT**: General conversation (e.g., "Hello", "How are you?")

The router uses TF-IDF classification with confidence scores. Low-confidence predictions fall back to CHAT mode.

## Technical Details

- **Vector Store**: Chroma with persistent storage
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **LLM**: Qwen2.5-1.5B-Instruct (GPU) or tinylama-1.1B-chat (CPU fallback)
- **Text Splitting**: RecursiveCharacterTextSplitter (1200 chars, 200 overlap)
- **Retrieval**: Top-5 most relevant chunks per query
- **RL Learning**: Confidence weight adjustments based on reward/penalty signals

## Project Structure

```
cortex/
├── embeddings.py      # Embedding model initialization
├── ingest.py          # Document ingestion pipeline
├── llm.py            # LLM setup and configuration
├── query.py          # RAG chain and query execution
├── router.py         # TF-IDF query router
├── memory.py         # Conversation history
├── persona.py        # System prompts
├── streaming.py      # Token streaming handler
├── ui.py             # Rich interactive CLI
├── cli.py            # Minimal CLI interface
├── logo.py           # ASCII logo generation
├── rl_router.py      # Reinforcement learning router
├── rl_cli.py         # RL training interface
└── rl_dashboard.py   # RL visualization dashboard
```

## Requirements

See `requirements.txt` for full dependencies. Key libraries:
- langchain, langchain-community, langchain-chroma
- chromadb
- transformers, torch
- rich (for UI)
- scikit-learn, joblib (for routing)