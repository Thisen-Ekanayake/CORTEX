# llm.py Documentation

## Overview

This module provides a unified interface for loading and configuring Large Language Models (LLMs) with automatic GPU/CPU detection and fallback mechanisms. It supports both GGUF-quantized models via LlamaCpp and HuggingFace transformers, making it flexible for different hardware configurations.

## Architecture

```
get_llm() Called
    ↓
Check GPU Availability (CUDA)
    ↓
┌─────────────┬─────────────────┐
│ GPU Found   │ No GPU          │
├─────────────┼─────────────────┤
│ LlamaCpp    │ CPU Fallback?   │
│ (Qwen 1.5B) │ Yes → HF Pipeline│
│             │ No  → Raise Error│
└─────────────┴─────────────────┘
```

## Core Function

### `get_llm(streaming=False, callbacks=None, cpu_fallback=True)`

Returns an appropriate LLM instance based on hardware availability.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `streaming` | bool | `False` | Enable token-by-token streaming output |
| `callbacks` | list | `None` | List of LangChain callback handlers |
| `cpu_fallback` | bool | `True` | Use CPU model if GPU unavailable |

**Returns:**
- `LlamaCpp` instance if GPU available
- `HuggingFacePipeline` instance if CPU fallback enabled
- Raises `RuntimeError` if no GPU and `cpu_fallback=False`

**Usage Examples:**

```python
from cortex.llm import get_llm

# Basic usage
llm = get_llm()
response = llm.invoke("What is machine learning?")

# With streaming
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = get_llm(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Without CPU fallback (GPU required)
llm = get_llm(cpu_fallback=False)
```

## GPU Configuration (LlamaCpp)

### Current Model: Qwen2.5-1.5B-Instruct

**Model Path:** `models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-q4_0.gguf`

**Quantization:** Q4_0 (4-bit quantization)

**Specifications:**
- **Parameters:** 1.5 billion
- **Context Window:** 32,768 tokens (configured)
- **Quantization:** 4-bit (Q4_0)
- **Memory Usage:** ~1-2GB VRAM
- **Speed:** Fast inference on consumer GPUs

### LlamaCpp Parameters

```python
LlamaCpp(
    model_path="models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-q4_0.gguf",
    n_ctx=32768,          # Context window size
    n_batch=64,           # Batch size for processing
    max_tokens=512,       # Maximum tokens to generate
    temperature=0.2,      # Sampling temperature
    n_threads=8,          # CPU threads for computation
    streaming=False,      # Enable streaming
    callbacks=[],         # Callback handlers
    verbose=False         # Suppress debug output
)
```

### Hyperparameter Tuning Guide

#### 1. **n_ctx** (Context Window)

**Current Value:** 32768

**What it does:** Maximum number of tokens the model can process (input + output)

**Tuning Guidelines:**

| Value | Use Case | Memory Impact | Speed Impact |
|-------|----------|---------------|--------------|
| 2048 | Short conversations | Low | Fast |
| 4096 | Standard conversations | Medium | Medium |
| 8192 | Long documents | High | Slower |
| 16384-32768 | Very long context | Very High | Much slower |

**Recommendations:**
```python
# For chatbots
n_ctx=4096

# For RAG systems
n_ctx=8192

# For document analysis
n_ctx=16384

# Maximum context (if memory allows)
n_ctx=32768
```

**Memory Calculation:**
- Each token uses ~1 byte in Q4 quantization
- n_ctx=32768 ≈ 32KB per context
- Batch processing multiplies this

#### 2. **n_batch** (Batch Size)

**Current Value:** 64

**What it does:** Number of tokens processed simultaneously during prompt evaluation

**Tuning Guidelines:**

| Value | Speed | Memory Usage | Best For |
|-------|-------|--------------|----------|
| 8-32 | Slower | Low | Limited VRAM |
| 64-128 | Balanced | Medium | Most GPUs |
| 256-512 | Fastest | High | High-end GPUs |

**Recommendations:**
```python
# For 4GB VRAM
n_batch=32

# For 6-8GB VRAM
n_batch=64

# For 12GB+ VRAM
n_batch=128

# For 24GB+ VRAM
n_batch=256
```

#### 3. **max_tokens** (Generation Length)

**Current Value:** 512

**What it does:** Maximum number of tokens to generate in response

**Tuning Guidelines:**

| Value | Use Case | Response Length |
|-------|----------|-----------------|
| 128-256 | Short answers, chatbots | 1-2 paragraphs |
| 512-1024 | Standard responses | 3-5 paragraphs |
| 1024-2048 | Long-form content | Full articles |
| 2048+ | Very long generation | Essays, reports |

**Recommendations:**
```python
# For Q&A
max_tokens=256

# For explanations
max_tokens=512

# For creative writing
max_tokens=1024

# For comprehensive analysis
max_tokens=2048
```

#### 4. **temperature** (Sampling Temperature)

**Current Value:** 0.2

**What it does:** Controls randomness in generation (0.0 = deterministic, 2.0 = very random)

**Tuning Guidelines:**

| Value | Behavior | Best For |
|-------|----------|----------|
| 0.0-0.3 | Very deterministic | Factual answers, code, analysis |
| 0.4-0.7 | Balanced | General conversation |
| 0.8-1.0 | Creative | Storytelling, brainstorming |
| 1.1-2.0 | Very creative | Experimental, artistic |

**Recommendations:**
```python
# For RAG/Q&A systems (factual)
temperature=0.1

# For chatbots (balanced)
temperature=0.5

# For creative writing
temperature=0.9

# For code generation
temperature=0.0
```

#### 5. **n_threads** (CPU Threads)

**Current Value:** 8

**What it does:** Number of CPU threads used for computation (even with GPU)

**Tuning Guidelines:**
```python
# Get optimal thread count
import os
cpu_count = os.cpu_count()

# General recommendation
n_threads = cpu_count - 2  # Leave some for system

# For 4-core CPU
n_threads=4

# For 8-core CPU
n_threads=6

# For 16-core CPU
n_threads=12
```

#### 6. **Additional Parameters** (Not Currently Used)

```python
LlamaCpp(
    # Current parameters...
    
    # Sampling parameters
    top_p=0.95,              # Nucleus sampling
    top_k=40,                # Top-k sampling
    repeat_penalty=1.1,      # Penalize repetition
    
    # Generation control
    stop=["</s>", "User:"],  # Stop sequences
    
    # Performance
    n_gpu_layers=35,         # Layers to offload to GPU (-1 = all)
    use_mlock=True,          # Lock model in RAM
    use_mmap=True,           # Use memory mapping
    
    # Advanced
    rope_freq_base=10000,    # RoPE frequency base
    rope_freq_scale=1.0,     # RoPE frequency scale
)
```

## Alternative GGUF Models

### Quantization Levels

**Q4_0** (Current) - 4-bit quantization
- **Size:** Smallest
- **Quality:** Good
- **Speed:** Fastest
- **Best for:** Consumer GPUs, laptops

**Q5_0** - 5-bit quantization
```python
model_path="models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-q5_0.gguf"
```
- **Size:** Medium
- **Quality:** Better
- **Speed:** Slower than Q4
- **Best for:** Mid-range GPUs

**Q6_K** - 6-bit quantization
```python
model_path="models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-q6_k.gguf"
```
- **Size:** Larger
- **Quality:** Very good
- **Speed:** Slower
- **Best for:** High-end GPUs

**Q8_0** - 8-bit quantization
```python
model_path="models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-q8_0.gguf"
```
- **Size:** Large
- **Quality:** Near full precision
- **Speed:** Slowest quantized
- **Best for:** Maximum quality

**F16** - Full 16-bit
```python
model_path="models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-f16.gguf"
```
- **Size:** Largest
- **Quality:** Maximum
- **Speed:** Requires high-end GPU

### Alternative Model Sizes

#### Smaller Models (Faster, Less Capable)

**Qwen2.5-0.5B**
```python
model_path="models/Qwen2.5-0.5B_Instruct/qwen2.5-0.5b-instruct-q4_0.gguf"
```
- **Parameters:** 500M
- **VRAM:** ~500MB
- **Best for:** Very limited hardware

**TinyLlama-1.1B**
```python
model_path="models/TinyLlama-1.1B/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```
- **Parameters:** 1.1B
- **VRAM:** ~800MB
- **Best for:** Quick responses, limited resources

#### Medium Models (Balanced)

**Qwen2.5-3B**
```python
model_path="models/Qwen2.5-3B_Instruct/qwen2.5-3b-instruct-q4_0.gguf"
```
- **Parameters:** 3B
- **VRAM:** ~2-3GB
- **Best for:** Good balance of speed and quality

**Phi-3-Mini-4K**
```python
model_path="models/Phi-3-mini-4k-instruct/phi-3-mini-4k-instruct-q4.gguf"
```
- **Parameters:** 3.8B
- **VRAM:** ~2-3GB
- **Best for:** High quality on small hardware

**Llama-3.2-3B**
```python
model_path="models/Llama-3.2-3B-Instruct/llama-3.2-3b-instruct-q4_0.gguf"
```
- **Parameters:** 3B
- **VRAM:** ~2-3GB
- **Best for:** Meta's latest small model

#### Larger Models (Better Quality, More Resources)

**Qwen2.5-7B**
```python
model_path="models/Qwen2.5-7B_Instruct/qwen2.5-7b-instruct-q4_0.gguf"
n_ctx=32768
```
- **Parameters:** 7B
- **VRAM:** ~4-5GB (Q4_0)
- **Best for:** High quality responses

**Llama-3.1-8B**
```python
model_path="models/Llama-3.1-8B-Instruct/llama-3.1-8b-instruct-q4_0.gguf"
n_ctx=131072  # 128K context
```
- **Parameters:** 8B
- **VRAM:** ~5-6GB (Q4_0)
- **Best for:** Very long context

**Mistral-7B**
```python
model_path="models/Mistral-7B-Instruct-v0.3/mistral-7b-instruct-v0.3.Q4_0.gguf"
```
- **Parameters:** 7B
- **VRAM:** ~4-5GB (Q4_0)
- **Best for:** Strong general performance

#### Very Large Models (Best Quality, High Requirements)

**Qwen2.5-14B**
```python
model_path="models/Qwen2.5-14B_Instruct/qwen2.5-14b-instruct-q4_0.gguf"
```
- **Parameters:** 14B
- **VRAM:** ~8-10GB (Q4_0)
- **Best for:** Professional applications

**Llama-3.1-70B** (Requires high-end GPU)
```python
model_path="models/Llama-3.1-70B-Instruct/llama-3.1-70b-instruct-q4_0.gguf"
n_gpu_layers=40  # Adjust based on VRAM
```
- **Parameters:** 70B
- **VRAM:** ~35-40GB (Q4_0)
- **Best for:** Maximum quality

## CPU Fallback Configuration

### Current Model: TinyLlama-1.1B-Chat

**Specifications:**
- **Parameters:** 1.1 billion
- **Memory:** ~2-4GB RAM
- **Speed:** Slow on CPU (1-5 tokens/sec)
- **Quality:** Basic conversational ability

### Alternative CPU Models

#### Via HuggingFace Transformers

**Qwen2.5-0.5B-Instruct**
```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # CPU uses float32
    low_cpu_mem_usage=True
)
```
- Faster than TinyLlama on CPU
- Better quality
- Smaller memory footprint

**Phi-2**
```python
model_name = "microsoft/phi-2"
```
- 2.7B parameters
- High quality for size
- Slower on CPU

**StableLM-2-1.6B**
```python
model_name = "stabilityai/stablelm-2-1_6b"
```
- Good balance of speed and quality
- Efficient on CPU

#### CPU-Optimized Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 8-bit quantization for CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)
```

## Alternative LLM Providers

### 1. **OpenAI API** (Recommended for Production)

```python
from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",  # or gpt-4o, gpt-4-turbo
        temperature=0.2,
        max_tokens=512,
        streaming=True
    )
```

**Pros:**
- Best quality
- Fast inference
- Reliable infrastructure
- No local resources

**Cons:**
- Requires API key
- Costs money
- Internet dependency
- Data privacy concerns

**Cost Comparison:**
- gpt-4o-mini: $0.15/1M input tokens
- gpt-4o: $2.50/1M input tokens
- gpt-4-turbo: $10/1M input tokens

### 2. **Anthropic Claude**

```python
from langchain_anthropic import ChatAnthropic

def get_llm():
    return ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.2,
        max_tokens=512,
        streaming=True
    )
```

**Best for:** Complex reasoning, safety-critical applications

### 3. **Google Gemini**

```python
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # or gemini-1.5-pro
        temperature=0.2,
        max_tokens=512
    )
```

**Best for:** Multimodal tasks, long context (1M+ tokens)

### 4. **Ollama** (Local, Easy Setup)

```python
from langchain_ollama import ChatOllama

def get_llm():
    return ChatOllama(
        model="qwen2.5:1.5b",  # or llama3.1, mistral, etc.
        temperature=0.2,
        num_predict=512
    )
```

**Pros:**
- Very easy setup
- Auto-downloads models
- Clean API
- Free and local

**Cons:**
- Requires Ollama service running
- Less control than LlamaCpp

**Installation:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull qwen2.5:1.5b

# Run service
ollama serve
```

### 5. **HuggingFace Inference API**

```python
from langchain_huggingface import HuggingFaceEndpoint

def get_llm():
    return HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token="your-token"
    )
```

**Best for:** Access to largest models without local resources

### 6. **LM Studio** (GUI Alternative)

```python
from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="local-model",
        temperature=0.2
    )
```

**Best for:** Non-technical users, visual model management

## Enhanced `get_llm()` Implementation

### With Advanced Configuration

```python
import torch
from langchain_community.llms.llamacpp import LlamaCpp
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

def get_llm(
    streaming=False,
    callbacks=None,
    cpu_fallback=True,
    model_size="1.5b",
    quantization="q4_0",
    temperature=0.2,
    max_tokens=512,
    context_window=8192
):
    """
    Get LLM with flexible configuration.
    
    Args:
        streaming: Enable token streaming
        callbacks: LangChain callback handlers
        cpu_fallback: Use CPU model if GPU unavailable
        model_size: "0.5b", "1.5b", "3b", "7b", etc.
        quantization: "q4_0", "q5_0", "q6_k", "q8_0"
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum generation length
        context_window: Context window size
    """
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        # Construct model path
        model_path = f"models/Qwen2.5-{model_size.upper()}_Instruct/qwen2.5-{model_size}-instruct-{quantization}.gguf"
        
        # Adjust batch size based on VRAM
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        n_batch = min(128, int(vram_gb * 16))  # Scale with VRAM
        
        # Calculate GPU layers based on model size
        layer_map = {
            "0.5b": 26, "1.5b": 28, "3b": 36,
            "7b": 32, "14b": 40, "32b": 64
        }
        n_gpu_layers = layer_map.get(model_size, -1)
        
        return LlamaCpp(
            model_path=model_path,
            n_ctx=context_window,
            n_batch=n_batch,
            max_tokens=max_tokens,
            temperature=temperature,
            n_threads=os.cpu_count() - 2,
            n_gpu_layers=n_gpu_layers,
            streaming=streaming,
            callbacks=callbacks or [],
            verbose=False,
            use_mlock=True,
            use_mmap=True
        )
    
    elif cpu_fallback:
        print("⚠️  No GPU detected, using CPU fallback (slower)")
        
        # Choose smaller model for CPU
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0
        )
    
    else:
        raise RuntimeError(
            "No GPU detected and CPU fallback disabled. "
            "Either enable GPU or set cpu_fallback=True"
        )
```

### With Model Auto-Download

```python
def get_llm(streaming=False, callbacks=None, cpu_fallback=True):
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        model_path = "models/Qwen2.5-1.5B_Instruct/qwen2.5-1.5b-instruct-q4_0.gguf"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Please download from: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF")
            
            # Alternative: Auto-download (requires huggingface_hub)
            from huggingface_hub import hf_hub_download
            
            print("Downloading model...")
            model_path = hf_hub_download(
                repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                filename="qwen2.5-1.5b-instruct-q4_0.gguf",
                local_dir="models/Qwen2.5-1.5B_Instruct"
            )
        
        return LlamaCpp(
            model_path=model_path,
            n_ctx=32768,
            n_batch=64,
            max_tokens=512,
            temperature=0.2,
            n_threads=8,
            streaming=streaming,
            callbacks=callbacks or [],
            verbose=False
        )
    
    elif cpu_fallback:
        # CPU implementation...
        pass
```

### With Fallback Chain

```python
def get_llm(streaming=False, callbacks=None):
    """Try GPU → CPU → API fallback."""
    
    # Try GPU
    if torch.cuda.is_available():
        try:
            return get_gpu_llm(streaming, callbacks)
        except Exception as e:
            print(f"GPU initialization failed: {e}")
    
    # Try CPU
    try:
        return get_cpu_llm(streaming, callbacks)
    except Exception as e:
        print(f"CPU initialization failed: {e}")
    
    # Fallback to API
    print("Using OpenAI API as fallback")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", streaming=streaming)
```

## Performance Optimization

### 1. **GPU Memory Management**

```python
# Clear CUDA cache before loading
import torch
torch.cuda.empty_cache()

# Monitor VRAM usage
def get_vram_usage():
    return torch.cuda.memory_allocated() / 1e9  # GB
```

### 2. **Optimal Layer Offloading**

```python
# Offload optimal layers based on VRAM
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

if vram_gb >= 8:
    n_gpu_layers = -1  # All layers
elif vram_gb >= 6:
    n_gpu_layers = 30
elif vram_gb >= 4:
    n_gpu_layers = 20
else:
    n_gpu_layers = 10
```

### 3. **Batch Processing**

```python
# Process multiple prompts efficiently
prompts = ["prompt1", "prompt2", "prompt3"]

llm = get_llm()
responses = llm.batch(prompts)
```

### 4. **Caching**

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())

llm = get_llm()
# Repeated queries will use cache
```

## Common Issues and Solutions

### Issue 1: Out of VRAM

**Symptoms:** CUDA out of memory error

**Solutions:**
```python
# Use smaller model
model_path="models/Qwen2.5-0.5B_Instruct/qwen2.5-0.5b-instruct-q4_0.gguf"

# Reduce context window
n_ctx=4096

# Reduce batch size
n_batch=32

# Offload fewer layers
n_gpu_layers=20
```

### Issue 2: Slow Generation

**Symptoms:** Very slow token generation

**Solutions:**
```python
# Increase batch size (if VRAM allows)
n_batch=128

# Use better quantization
quantization="q4_0"  # Fastest

# Offload all layers to GPU
n_gpu_layers=-1

# Reduce context window
n_ctx=4096
```

### Issue 3: Poor Quality Responses

**Symptoms:** Incoherent or low-quality outputs

**Solutions:**
```python
# Use larger model
model_size="7b"

# Use better quantization
quantization="q6_k"  # or q8_0

# Adjust temperature
temperature=0.5  # Increase for more creativity

# Increase max tokens
max_tokens=1024

# Add system prompt
# (in your application code)
```

### Issue 4: Model Not Found

**Symptoms:** FileNotFoundError

**Solutions:**
```bash
# Download from HuggingFace
# https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF

# Or use huggingface-cli
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_0.gguf \
  --local-dir models/Qwen2.5-1.5B_Instruct
```

### Issue 5: CPU Fallback Too Slow

**Symptoms:** Generation takes minutes

**Solutions:**
```python
# Use smaller model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Reduce max tokens
max_new_tokens=128

# Or use API instead
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
```

## Model Selection Guide

| Hardware | Recommended Model | Quantization | VRAM | Speed |
|----------|------------------|--------------|------|-------|
| No GPU | Qwen2.5-0.5B (CPU) | - | 2GB RAM | Slow |
| 4GB GPU | Qwen2.5-1.5B | Q4_0 | 1-2GB | Fast |
| 6GB GPU | Qwen2.5-3B | Q4_0 | 2-3GB | Fast |
| 8GB GPU | Qwen2.5-7B | Q4_0/Q5_0 | 4-5GB | Medium |
| 12GB GPU | Qwen2.5-7B | Q6_K/Q8_0 | 6-8GB | Medium |
| 16GB GPU | Qwen2.5-14B | Q4_0 | 8-10GB | Medium |
| 24GB+ GPU | Qwen2.5-32B+ | Q4_0+ | 15-20GB | Slower |

## Best Practices

1. **Always check GPU availability** before initializing models
2. **Use appropriate quantization** - Q4_0 for most cases, higher for quality
3. **Monitor VRAM usage** to prevent OOM errors
4. **Set reasonable context windows** - larger isn't always better
5. **Implement proper error handling** for model loading failures
6. **Cache model instances** - don't reload for every request
7. **Use streaming** for better user experience in chat applications
8. **Profile performance** on your hardware before production
9. **Have fallback options** for different hardware configurations
10. **Keep models updated** - newer versions often have better quality

## Usage Examples

### Basic RAG Application

```python
from cortex.llm import get_llm
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from cortex.embeddings import get_embeddings

# Load LLM
llm = get_llm(temperature=0.1)

# Load vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=get_embeddings()
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query
response = qa_chain.invoke("What is machine learning?")
print(response)
```

### Streaming Chat

```python
from cortex.llm import get_llm
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize with streaming
llm = get_llm(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Chat
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    
    print("Assistant: ", end="")
    llm.invoke(query)
    print()
```

### With Custom Callbacks

```python
from langchain.callbacks.base import BaseCallbackHandler

class TokenCounterCallback(BaseCallbackHandler):
    def __init__(self):
        self.token_count = 0
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.token_count += 1

counter = TokenCounterCallback()
llm = get_llm(streaming=True, callbacks=[counter])

response = llm.invoke("Explain quantum computing")
print(f"\nGenerated {counter.token_count} tokens")
```

### Multi-turn Conversation

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = get_llm(temperature=0.7)

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat
conversation.predict(input="Hi, I'm learning about AI")
conversation.predict(input="What topics should I focus on?")
conversation.predict(input="Tell me more about the second topic")
```

## Dependencies

```txt
# Core dependencies
torch>=2.0.0
langchain>=0.1.0
langchain-community>=0.0.1
transformers>=4.30.0

# For LlamaCpp
llama-cpp-python>=0.2.0

# Optional for API providers
langchain-openai>=0.0.1
langchain-anthropic>=0.0.1
langchain-google-genai>=0.0.1
langchain-ollama>=0.0.1

# For model downloads
huggingface-hub>=0.16.0
```

### Installation

```bash
# Basic installation
pip install torch langchain langchain-community transformers

# LlamaCpp with GPU support (CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# LlamaCpp with Metal support (Mac)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# LlamaCpp CPU only
pip install llama-cpp-python

# All optional dependencies
pip install langchain-openai langchain-anthropic langchain-google-genai
```

## Model Downloads

### HuggingFace GGUF Models

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_0.gguf \
  --local-dir models/Qwen2.5-1.5B_Instruct

# Other popular models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_0.gguf \
  --local-dir models/Qwen2.5-7B_Instruct

huggingface-cli download bartowski/Llama-3.1-8B-Instruct-GGUF \
  Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir models/Llama-3.1-8B-Instruct
```

### Directory Structure

```
project/
├── models/
│   ├── Qwen2.5-1.5B_Instruct/
│   │   └── qwen2.5-1.5b-instruct-q4_0.gguf
│   ├── Qwen2.5-7B_Instruct/
│   │   └── qwen2.5-7b-instruct-q4_0.gguf
│   └── ...
├── cortex/
│   ├── llm.py
│   ├── embeddings.py
│   └── ingest.py
└── main.py
```

## Further Reading

- [LlamaCpp Documentation](https://python.langchain.com/docs/integrations/llms/llamacpp)
- [LlamaCpp GitHub](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Explanation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [HuggingFace GGUF Models](https://huggingface.co/models?library=gguf)
- [Quantization Guide](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)
- [LangChain LLM Integration](https://python.langchain.com/docs/integrations/llms/)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen)
- [Model Benchmarks](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)