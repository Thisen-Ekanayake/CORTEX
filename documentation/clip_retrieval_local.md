# CLIP Local Image Retrieval Documentation

## Overview

This module enables **semantic image search** using OpenAI's CLIP model. It allows you to search local image collections using natural language queries (e.g., "a sunset over mountains") and returns the most visually similar images.

## What It Does

1. **Indexes images**: Converts all images in a directory into 512-dimensional vector embeddings
2. **Processes text queries**: Converts your text prompt into the same vector space
3. **Finds matches**: Computes cosine similarity between text and image embeddings
4. **Returns top results**: Ranks images by relevance and returns top-k matches

## Key Components

### Model Loading (Lines 7-15)
```python
model = CLIPModel.from_pretrained(
    "/ml/CORTEX/models/CLIP-vit-large-patch14",
    local_files_only=True
)
```

**What**: Loads CLIP-ViT-Large-Patch14 model from local storage
**Why**: 
- Loaded once at module level for performance (avoid reloading on every query)
- `local_files_only=True` prevents downloading, uses cached model
- GPU-accelerated if available (`cuda`), falls back to CPU

### `index_local_images(dir_path)` (Lines 17-37)

**What**: Converts all images in a directory to embeddings

**Process**:
1. Iterate through directory files
2. Load each image as RGB
3. Convert to CLIP embedding (512-dim vector)
4. **L2 normalize** embeddings (critical for cosine similarity)
5. Return stacked embeddings tensor + file paths

**Returns**: `(embeddings_tensor, paths_list)` or `(None, [])`

### `retrieve_local(prompt, embeddings, paths, top_k)` (Lines 39-54)

**What**: Searches images using text query

**Process**:
1. Encode text prompt to embedding
2. L2 normalize text embedding
3. Compute similarity: `embeddings @ text_embedding.T` (matrix multiplication = cosine similarity)
4. Get top-k highest scores
5. Return `[(path, score), ...]` sorted by relevance

**Similarity Scores**: Range from -1 to 1 (typically 0.15-0.35 for matches)

### CLI Interface (Lines 56-82)

**Usage**:
```bash
python clip_retrieval_local.py \
    --prompt "a cat sleeping on a couch" \
    --images_dir "data/images" \
    --top_k 5
```

**Output**:
```
0.342 — data/images/cat_sofa.jpg
0.318 — data/images/tabby_sleeping.jpg
0.287 — data/images/kitten_nap.jpg
```

## How It Works: CLIP Embedding Space

```
Text: "a dog"     → [0.12, -0.45, 0.89, ...] (512-dim)
Image: dog.jpg    → [0.15, -0.42, 0.91, ...] (512-dim)
                      ↓
           Cosine Similarity = 0.92 (high match!)

Text: "a cat"     → [-0.31, 0.67, -0.12, ...]
Image: dog.jpg    → [0.15, -0.42, 0.91, ...]
                      ↓
           Cosine Similarity = 0.18 (low match)
```

**Key Insight**: CLIP maps semantically similar images and text to nearby points in vector space, enabling zero-shot retrieval without training on your specific images.

## Alternative Approaches

### 1. **Different CLIP Models**

| Model | Size | Speed | Accuracy | Use When |
|-------|------|-------|----------|----------|
| ViT-B/32 | 150MB | Fast | Good | Quick prototyping, limited resources |
| ViT-B/16 | 350MB | Medium | Better | Balanced production use |
| **ViT-L/14** (current) | 890MB | Slow | Best | Accuracy is priority |
| ResNet-50 | 150MB | Fast | Good | Legacy compatibility |

**Switch models**:
```python
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
```

### 2. **FAISS for Large-Scale Search**

**When**: >100k images  
**Why**: Approximate nearest neighbor (ANN) search is 100x faster  
**How**:
```python
import faiss
index = faiss.IndexFlatIP(512)  # Inner product index
index.add(embeddings.numpy())
scores, indices = index.search(query_embedding, k=5)
```

### 3. **Sentence-Transformers CLIP**

**When**: Want simpler API  
**Why**: Less boilerplate code  
**How**:
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('clip-ViT-B-32')
image_emb = model.encode(Image.open('photo.jpg'))
text_emb = model.encode("a cat")
similarity = util.cos_sim(image_emb, text_emb)
```

### 4. **OpenCLIP**

**When**: Need newer/more models  
**Why**: More model variants, better performance on some tasks  
**How**:
```python
import open_clip
model, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
```

### 5. **Vector Databases**

**For production with millions of images**:
- **Pinecone**: Managed cloud vector DB
- **Weaviate**: Open-source with hybrid search
- **Milvus**: Self-hosted, GPU-accelerated
- **Qdrant**: Rust-based, very fast

### 6. **Traditional Methods**

**For simple visual similarity (no semantic understanding)**:
- **Perceptual hashing** (pHash): Duplicate detection
- **Feature extraction + KNN**: SIFT/SURF features
- **Reverse image search APIs**: Google Vision, TinEye

## Performance Optimization

### Speed Up Indexing (Batch Processing)
```python
# Current: ~2 images/sec
# Batched: ~20 images/sec on GPU

def index_batched(dir_path, batch_size=32):
    # Load multiple images at once
    inputs = processor(images=batch_images, return_tensors="pt")
    embeddings = model.get_image_features(**inputs)
```

### Cache Embeddings
```python
# Save after first indexing
torch.save({'embeddings': emb, 'paths': paths}, 'cache.pt')

# Load on subsequent runs
cached = torch.load('cache.pt')
```

### GPU Optimization
```python
# Enable TF32 on Ampere GPUs (30% faster)
torch.backends.cuda.matmul.allow_tf32 = True
```

## Common Use Cases

1. **Photo management**: Search personal photo libraries by description
2. **E-commerce**: Find products by visual description
3. **Digital asset management**: Locate stock photos/graphics
4. **Content moderation**: Find inappropriate images
5. **Research**: Analyze image datasets

## Limitations

- **Not exact match**: Returns semantic similarity, not pixel-perfect duplicates
- **Biased**: Reflects training data biases (LAION-400M dataset)
- **English-centric**: Works best with English queries
- **Resource intensive**: Large model requires significant RAM/VRAM
- **Static index**: Need to re-index when adding images

## Quick Start Example

```python
from clip_retrieval_local import index_local_images, retrieve_local

# Index once
embeddings, paths = index_local_images("my_photos")

# Search multiple times
results = retrieve_local("sunset at the beach", embeddings, paths, top_k=5)
for path, score in results:
    print(f"{score:.3f} - {path}")
```

## Conclusion

**Best for**: Local image search with <100k images, privacy-sensitive applications, offline usage

**Consider alternatives when**: Need real-time search on millions of images, cloud deployment, or multi-language support