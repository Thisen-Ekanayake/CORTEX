from langchain.tools import tool
from image_search.clip_retrieval_local import (
    index_local_images,
    retrieve_local
)
from image_search.google_search import fetch_google_images
from image_search.realtime_retrieval import search_and_retrieve

IMAGE_DIR = "data/images"

_image_embeddings = None
_image_paths = None

def _ensure_index():
    """
    Ensure local image index is loaded (lazy initialization).
    
    Loads and indexes all images from IMAGE_DIR if not already indexed.
    Uses global variables to cache the embeddings and paths.
    """
    global _image_embeddings, _image_paths
    if _image_embeddings is None:
        _image_embeddings, _image_paths = index_local_images(IMAGE_DIR)

@tool("local_image_search")
def local_image_search(query: str, top_k: int = 5):
    """
    Retrieve relevant local images using CLIP embeddings.
    
    Searches through pre-indexed local images and returns the top-k most
    similar images based on CLIP text-image similarity.
    
    Args:
        query: Text query describing desired images.
        top_k: Number of top results to return (default: 5).
    
    Returns:
        list: List of dicts with "path" and "score" keys for each result.
    """
    _ensure_index()

    results = retrieve_local(
        prompt=query,
        image_embeddings=_image_embeddings,
        image_paths=_image_paths,
        top_k=top_k
    )

    return [
        {"path": path, "score": score}
        for path, score in results
    ]

@tool("google_image_search")
def google_image_search(query: str, num_images: int = 10):
    """
    Fetch image URLs from Google Images search.
    
    Uses Google Custom Search API to find images matching the query.
    Only returns JPG/PNG images with safe search enabled.
    
    Args:
        query: Search query string.
        num_images: Number of image URLs to fetch (default: 10).
    
    Returns:
        list: List of dicts with "url" key for each image URL.
    """
    urls = fetch_google_images(query, num_images=num_images)

    return [{"url": u} for u in urls]

@tool("realtime_image_retrieval")
def realtime_image_retrieval(query: str, num_images: int = 20, top_k: int = 5):
    """
    Search web images, download them, and rerank using CLIP.
    
    Complete pipeline: searches Google Images, downloads images to cache,
    indexes with CLIP, and returns top-k most similar images.
    
    Args:
        query: Text query for image search.
        num_images: Number of images to fetch from Google (default: 20).
        top_k: Number of top results after CLIP ranking (default: 5).
    
    Returns:
        list: List of dicts with "path" and "score" keys for top matches.
    """
    results = search_and_retrieve(
        prompt=query,
        num_images=num_images,
        top_k=top_k
    )

    return [
        {"path": path, "score": score}
        for path, score in results
    ]