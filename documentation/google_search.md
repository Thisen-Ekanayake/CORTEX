# Google Images Fetch Documentation

## Overview

This module fetches image URLs from Google Images using the Google Custom Search API. It provides a simple CLI interface to search for images and retrieve their URLs for downloading or processing.

## What It Does

1. **Searches Google Images** using a text query
2. **Retrieves image URLs** (not actual image files)
3. **Filters results** by file type (JPG/PNG) and safety settings
4. **Returns up to 100 URLs** per query (API limit)

## Key Components

### Environment Setup (Lines 1-11)
```python
API_KEY = os.environ["GOOGLE_API_KEY"]
CX = os.environ["CX"]  # Custom Search Engine ID
```

**Required**: `.env` file with:
```
GOOGLE_API_KEY=your_api_key_here
CX=your_custom_search_engine_id
```

**Getting Credentials**:
1. **API Key**: Get from [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. **CX (Search Engine ID)**: Create at [Programmable Search Engine](https://programmablesearchengine.google.com/)

### `fetch_google_images(prompt, num_images=10)` (Lines 15-27)

**Parameters**:
- `prompt` (str): Search query (e.g., "sunset over mountains")
- `num_images` (int): Number of results (max 100 per query, default 10)

**Returns**: List of image URL strings

**Search Parameters**:
- `fileType`: "jpg|png" - Only JPEG and PNG images
- `safe`: "active" - Filters explicit content
- `num`: Number of results to return

### CLI Interface (Lines 29-48)

**Usage**:
```bash
python fetch_google_images.py \
    --prompt "golden retriever puppy" \
    --num_images 20
```

**Output**:
```
https://example.com/image1.jpg
https://example.com/image2.png
https://example.com/image3.jpg
...
```

## API Limits & Costs

| Tier | Daily Limit | Cost |
|------|-------------|------|
| **Free** | 100 queries/day | $0 |
| **Paid** | 10,000 queries/day | $5 per 1,000 queries |

**Note**: Each query can return up to 10 results. For 100 images, you need 10 queries.

## Alternative Approaches

### 1. **Bing Image Search API**

**Pros**:
- More generous free tier (1,000 queries/month)
- Faster response times
- Better for commercial use

**Cons**:
- Requires Azure account
- Different result quality

**Implementation**:
```python
from azure.cognitiveservices.search.imagesearch import ImageSearchClient
from msrest.authentication import CognitiveServicesCredentials

client = ImageSearchClient(
    endpoint="https://api.bing.microsoft.com",
    credentials=CognitiveServicesCredentials(api_key)
)

results = client.images.search(query="cats", count=10)
urls = [img.content_url for img in results.value]
```

### 2. **Unsplash API**

**Pros**:
- High-quality, free-to-use images
- 5,000 requests/hour (free tier)
- No copyright issues
- Direct download URLs

**Cons**:
- Smaller dataset than Google
- Photography-focused (not general images)

**Implementation**:
```python
import requests

def fetch_unsplash_images(query, num=10):
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {"query": query, "per_page": num}
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    return [photo["urls"]["regular"] for photo in data["results"]]
```

### 3. **Pexels API**

**Pros**:
- Free, high-quality stock photos/videos
- 200 requests/hour, unlimited requests/month
- No attribution required

**Cons**:
- Curated collection (smaller than Google)
- Stock photography style

**Implementation**:
```python
import requests

def fetch_pexels_images(query, num=10):
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": num}
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    return [photo["src"]["large"] for photo in data["photos"]]
```

### 4. **Flickr API**

**Pros**:
- Massive image database
- License filtering (Creative Commons)
- Free tier: 3,600 queries/hour

**Cons**:
- More complex API
- Varying image quality

**Implementation**:
```python
import flickrapi

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
photos = flickr.photos.search(text=query, per_page=10)

urls = [
    f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
    for photo in photos['photos']['photo']
]
```

### 5. **Web Scraping (Not Recommended)**

**Pros**:
- No API costs
- Access to any website

**Cons**:
- **Violates Terms of Service**
- Legal risks
- Unreliable (sites change structure)
- Rate limiting/blocking

**Alternative**: If needed, use `selenium` or `playwright` with proper rate limiting, but check ToS first.

### 6. **DuckDuckGo Images (Free, No API Key)**

**Pros**:
- No API key required
- Completely free
- Privacy-focused
- Good for testing/prototyping

**Cons**:
- Unofficial API
- May break without notice
- Rate limiting

**Implementation**:
```python
from duckduckgo_search import DDGS

def fetch_duckduckgo_images(query, num=10):
    with DDGS() as ddgs:
        results = list(ddgs.images(query, max_results=num))
    return [r['image'] for r in results]
```

## Enhanced Features

### Download Images (Not Just URLs)

```python
import requests
from pathlib import Path

def download_images(urls, output_dir="downloads"):
    Path(output_dir).mkdir(exist_ok=True)
    
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Determine extension
            ext = url.split('.')[-1].split('?')[0]
            if ext not in ['jpg', 'jpeg', 'png']:
                ext = 'jpg'
            
            filepath = Path(output_dir) / f"image_{i+1}.{ext}"
            filepath.write_bytes(response.content)
            print(f"Downloaded: {filepath}")
            
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Usage
urls = fetch_google_images("cats", num_images=10)
download_images(urls)
```

### Image Size Filtering

```python
def fetch_google_images(prompt, num_images=10, size="medium"):
    """
    size: 'icon', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'huge'
    """
    search_params = {
        "q": prompt,
        "num": num_images,
        "fileType": "jpg|png",
        "safe": "active",
        "imgSize": size,  # Filter by size
    }
    
    gis.search(search_params=search_params)
    return [img.url for img in gis.results()]
```

### Color Filtering

```python
def fetch_google_images(prompt, num_images=10, color="any"):
    """
    color: 'color', 'gray', 'mono', 'red', 'orange', 'yellow', 
           'green', 'teal', 'blue', 'purple', 'pink', 'white', 
           'gray', 'black', 'brown'
    """
    search_params = {
        "q": prompt,
        "num": num_images,
        "imgDominantColor": color,
    }
    
    gis.search(search_params=search_params)
    return [img.url for img in gis.results()]
```

## Best Practices

### 1. Rate Limiting
```python
import time

def fetch_with_rate_limit(queries, delay=1.0):
    """Avoid hitting API limits"""
    all_urls = []
    for query in queries:
        urls = fetch_google_images(query)
        all_urls.extend(urls)
        time.sleep(delay)  # Wait between requests
    return all_urls
```

### 2. Error Handling
```python
def fetch_google_images_safe(prompt, num_images=10):
    """With error handling"""
    try:
        search_params = {
            "q": prompt,
            "num": min(num_images, 100),  # Enforce API limit
            "fileType": "jpg|png",
            "safe": "active",
        }
        
        gis.search(search_params=search_params)
        results = [img.url for img in gis.results()]
        
        if not results:
            print(f"No results found for: {prompt}")
        
        return results
        
    except Exception as e:
        print(f"Error fetching images: {e}")
        return []
```

### 3. Caching Results
```python
import json
from pathlib import Path

def fetch_with_cache(prompt, num_images=10, cache_dir="cache"):
    """Cache results to avoid repeated API calls"""
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = Path(cache_dir) / f"{prompt.replace(' ', '_')}.json"
    
    # Check cache
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    
    # Fetch and cache
    urls = fetch_google_images(prompt, num_images)
    with open(cache_file, 'w') as f:
        json.dump(urls, f)
    
    return urls
```

## Comparison Table

| Service | Free Tier | Quality | Speed | Best For |
|---------|-----------|---------|-------|----------|
| **Google** | 100/day | High | Fast | General search |
| **Bing** | 1,000/month | High | Fast | Commercial use |
| **Unsplash** | 5,000/hour | Very High | Fast | Professional photos |
| **Pexels** | Unlimited | High | Fast | Stock photography |
| **Flickr** | 3,600/hour | Variable | Medium | Creative Commons |
| **DuckDuckGo** | Unlimited* | Medium | Fast | Testing/prototyping |

*Unofficial, may have hidden limits

## Quick Start

1. **Setup credentials**:
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_key" > .env
echo "CX=your_cx" >> .env
```

2. **Install dependencies**:
```bash
pip install google-images-search python-dotenv
```

3. **Run**:
```bash
python fetch_google_images.py --prompt "mountains" --num_images 20
```

## Limitations

- **API quota**: Free tier limited to 100 queries/day
- **URL only**: Returns URLs, not actual images
- **Dead links**: URLs may expire or become invalid
- **Copyright**: Images may be copyrighted (check usage rights)
- **Rate limits**: Aggressive usage may trigger throttling

## Conclusion

**Best for**: Quick prototyping, small-scale image collection, diverse image types

**Consider alternatives when**: 
- Need high-quality stock photos → Unsplash/Pexels
- High-volume production use → Bing API
- Free/unlimited usage → DuckDuckGo (with caution)
- Specific licenses needed → Flickr with CC filtering