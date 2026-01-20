# Hybrid Google + CLIP Image Search Documentation

## Overview

This module combines **Google Image Search** (broad web coverage) with **CLIP semantic retrieval** (intelligent re-ranking) to find the most relevant images for a text query. It downloads images from Google, then uses CLIP to select the best matches.

## What It Does

1. **Fetches images** from Google Images using text query
2. **Downloads images** to local cache directory
3. **Indexes with CLIP** to create semantic embeddings
4. **Re-ranks results** using CLIP similarity scores
5. **Returns top-k** most semantically relevant images

## Why This Approach?

**Problem**: Google returns many images, but relevance varies  
**Solution**: Use CLIP to intelligently filter and re-rank based on semantic similarity

**Example**:
```
Query: "a golden retriever playing in a park"

Google returns (20 images):
- 5 golden retrievers in parks ✓
- 3 golden retrievers indoors ✗
- 8 other dogs in parks ✗
- 4 random images ✗

CLIP re-ranks → Returns top 5 golden retrievers in parks ✓
```

## Key Components

### Cache Directory (Lines 7-8)
```python
CACHE_DIR = "downloaded_images"
os.makedirs(CACHE_DIR, exist_ok=True)
```
**Purpose**: Stores downloaded images locally to avoid re-downloading

### `download_images(urls)` (Lines 10-22)

**What**: Downloads images from URLs to cache directory

**Process**:
1. Try to download each URL
2. Extract file extension from URL
3. Save as `img_0.jpg`, `img_1.png`, etc.
4. Skip failed downloads (timeout, 404, etc.)

**Returns**: List of successfully saved file paths

### `search_and_retrieve(prompt, num_images=20, top_k=5)` (Lines 24-28)

**What**: Complete pipeline from query to results

**Pipeline**:
```
1. fetch_google_images(prompt, 20)
   ↓
2. download_images(urls)
   ↓
3. index_local_images(CACHE_DIR)
   ↓
4. retrieve_local(prompt, embeddings, paths, 5)
   ↓
5. Return top 5 most relevant images
```

**Parameters**:
- `prompt` (str): Search query
- `num_images` (int): Images to fetch from Google (default: 20)
- `top_k` (int): Final results to return (default: 5)

**Why fetch more than needed?**: Get 20 candidates, then CLIP selects best 5

### CLI Interface (Lines 30-62)

**Usage**:
```bash
python hybrid_search.py \
    --prompt "a sunset over the ocean" \
    --num_images 30 \
    --top_k 10
```

**Output**:
```
0.342 — downloaded_images/img_3.jpg
0.318 — downloaded_images/img_7.jpg
0.287 — downloaded_images/img_1.jpg
...
```

## How It Works: Two-Stage Ranking

### Stage 1: Google Search (Recall)
- **Goal**: Cast wide net, get diverse candidates
- **Method**: Keyword matching, PageRank, image metadata
- **Output**: 20-100 images with varying relevance

### Stage 2: CLIP Re-ranking (Precision)
- **Goal**: Find semantically best matches
- **Method**: Deep learning vision-language similarity
- **Output**: Top 5-10 most relevant images

**Analogy**:
```
Google = Librarian who finds all books with "cats" in title
CLIP = Expert who reads books and finds ones actually about cats
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Latency** | 10-30 seconds | Google API + downloads + CLIP |
| **Accuracy** | High | CLIP filters Google noise |
| **Cost** | $0.005/query | Google API pricing |
| **Cache hit** | 1-2 seconds | If images already downloaded |

**Breakdown**:
- Google API: 1-2 seconds
- Download 20 images: 5-10 seconds
- CLIP indexing: 3-5 seconds
- CLIP retrieval: <0.1 seconds

## Alternative Approaches

### 1. **Direct CLIP on Local Dataset**

**When**: You have pre-collected image dataset  
**Pros**: No API costs, instant results, offline  
**Cons**: Limited to your collection

```python
# Just use CLIP on existing images
embeddings, paths = index_local_images("my_dataset")
results = retrieve_local(prompt, embeddings, paths, top_k=5)
```

### 2. **Google Only (No CLIP Re-ranking)**

**When**: Speed > accuracy  
**Pros**: Faster (no CLIP processing), simpler  
**Cons**: Lower quality results

```python
urls = fetch_google_images(prompt, num_images=5)
paths = download_images(urls)
# Use first 5 directly without re-ranking
```

### 3. **CLIP-Retrieval with Laion-5B**

**When**: Need massive scale search  
**Pros**: 5 billion pre-indexed images  
**Cons**: Requires API/server, costs money

```python
from clip_retrieval.clip_client import ClipClient

client = ClipClient(url="https://knn.laion.ai/knn-service")
results = client.query(text=prompt, num_images=10)
```

### 4. **Unsplash API + CLIP**

**When**: Need high-quality stock photos  
**Pros**: Better image quality, free tier  
**Cons**: Smaller dataset than Google

```python
def search_and_retrieve_unsplash(prompt, num_images=20, top_k=5):
    # Fetch from Unsplash instead of Google
    urls = fetch_unsplash_images(prompt, num_images)
    paths = download_images(urls)
    emb, paths = index_local_images(CACHE_DIR)
    return retrieve_local(prompt, emb, paths, top_k)
```

### 5. **Multi-Source Aggregation**

**When**: Need diverse, comprehensive results  
**Pros**: Best coverage, quality diversity  
**Cons**: Slower, more API costs

```python
def multi_source_search(prompt, total=30, top_k=5):
    urls = []
    urls += fetch_google_images(prompt, 10)
    urls += fetch_unsplash_images(prompt, 10)
    urls += fetch_pexels_images(prompt, 10)
    
    paths = download_images(urls)
    emb, paths = index_local_images(CACHE_DIR)
    return retrieve_local(prompt, emb, paths, top_k)
```

### 6. **Streaming Search (Progressive Results)**

**When**: User needs fast initial results  
**Pros**: Better UX, perceived speed  
**Cons**: More complex implementation

```python
def streaming_search(prompt, num_images=20, top_k=5):
    urls = fetch_google_images(prompt, num_images)
    
    # Download and index in batches
    batch_size = 5
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        paths = download_images(batch_urls)
        
        # Yield intermediate results
        emb, paths = index_local_images(CACHE_DIR)
        results = retrieve_local(prompt, emb, paths, min(top_k, len(paths)))
        
        yield results  # Progressive output
```

## Enhanced Features

### 1. **Deduplication**

```python
from PIL import Image
import imagehash

def download_images_deduplicated(urls):
    """Avoid downloading duplicate images"""
    saved = []
    hashes = set()
    
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, timeout=10)
            img = Image.open(io.BytesIO(r.content))
            
            # Check perceptual hash
            img_hash = imagehash.phash(img)
            if img_hash in hashes:
                continue  # Skip duplicate
            
            hashes.add(img_hash)
            
            path = os.path.join(CACHE_DIR, f"img_{i}.jpg")
            img.save(path)
            saved.append(path)
            
        except Exception:
            continue
    
    return saved
```

### 2. **Persistent Cache**

```python
import hashlib
import json

def search_and_retrieve_cached(prompt, num_images=20, top_k=5):
    """Cache results to avoid re-downloading"""
    
    # Create cache key
    cache_key = hashlib.md5(f"{prompt}_{num_images}".encode()).hexdigest()
    cache_file = f"cache/{cache_key}.json"
    
    # Check cache
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached_paths = json.load(f)
        
        # Re-index cached images
        emb, paths = index_local_images(CACHE_DIR)
        return retrieve_local(prompt, emb, paths, top_k)
    
    # Perform search
    urls = fetch_google_images(prompt, num_images)
    paths = download_images(urls)
    
    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(paths, f)
    
    # Continue as normal
    emb, paths = index_local_images(CACHE_DIR)
    return retrieve_local(prompt, emb, paths, top_k)
```

### 3. **Image Validation**

```python
from PIL import Image

def download_images_validated(urls, min_size=200):
    """Only save images meeting size requirements"""
    saved = []
    
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, timeout=10)
            img = Image.open(io.BytesIO(r.content))
            
            # Check dimensions
            if img.width < min_size or img.height < min_size:
                continue  # Skip small images
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            path = os.path.join(CACHE_DIR, f"img_{i}.jpg")
            img.save(path, 'JPEG')
            saved.append(path)
            
        except Exception:
            continue
    
    return saved
```

### 4. **Batch Processing**

```python
def batch_search_queries(queries, num_images=20, top_k=5):
    """Process multiple queries efficiently"""
    results = {}
    
    for query in queries:
        print(f"Processing: {query}")
        query_results = search_and_retrieve(query, num_images, top_k)
        results[query] = query_results
    
    return results

# Usage
queries = ["sunset", "mountain landscape", "ocean waves"]
all_results = batch_search_queries(queries)
```

### 5. **Cleanup Old Cache**

```python
import time

def cleanup_old_cache(max_age_hours=24):
    """Remove images older than specified hours"""
    cutoff = time.time() - (max_age_hours * 3600)
    
    for fname in os.listdir(CACHE_DIR):
        fpath = os.path.join(CACHE_DIR, fname)
        
        if os.path.getmtime(fpath) < cutoff:
            os.remove(fpath)
            print(f"Deleted old cache: {fname}")
```

## Best Practices

### 1. **Optimize num_images vs top_k**

```python
# Good ratios:
search_and_retrieve(prompt, num_images=20, top_k=5)   # 4:1 ratio
search_and_retrieve(prompt, num_images=50, top_k=10)  # 5:1 ratio

# Poor ratios:
search_and_retrieve(prompt, num_images=10, top_k=10)  # No re-ranking benefit
search_and_retrieve(prompt, num_images=100, top_k=3)  # Wasteful
```

### 2. **Error Handling**

```python
def search_and_retrieve_safe(prompt, num_images=20, top_k=5):
    """With comprehensive error handling"""
    try:
        urls = fetch_google_images(prompt, num_images)
        
        if not urls:
            return []
        
        paths = download_images(urls)
        
        if len(paths) < top_k:
            print(f"Warning: Only {len(paths)} images downloaded")
        
        emb, paths = index_local_images(CACHE_DIR)
        
        if emb is None:
            return []
        
        return retrieve_local(prompt, emb, paths, min(top_k, len(paths)))
        
    except Exception as e:
        print(f"Error in search pipeline: {e}")
        return []
```

### 3. **Memory Management**

```python
import gc

def search_and_retrieve_efficient(prompt, num_images=20, top_k=5):
    """Clean up memory after processing"""
    results = search_and_retrieve(prompt, num_images, top_k)
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return results
```

## Limitations

- **Slow**: 10-30 seconds per query (API + downloads + CLIP)
- **API costs**: Google charges after free tier
- **Disk space**: Cache grows with usage
- **Dead links**: Some Google URLs may be invalid
- **Copyright**: Downloaded images may have restrictions

## Use Cases

1. **Visual research**: Quickly find reference images for creative work
2. **Dataset creation**: Build custom image datasets for ML
3. **Content discovery**: Find high-quality images for presentations
4. **Product search**: Find similar products across the web
5. **Reverse engineering**: Understand what images match a concept

## Conclusion

**Best for**: Finding high-quality, semantically relevant images when local dataset isn't available

**Consider alternatives when**:
- Speed critical → Use Google only (skip CLIP)
- Offline needed → Use pre-indexed local dataset
- Scale required → Use Laion-5B CLIP index
- Quality matters → Unsplash/Pexels + CLIP