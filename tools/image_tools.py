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
    global _image_embeddings, _image_paths
    if _image_embeddings is None:
        _image_embeddings, _image_paths = index_local_images(IMAGE_DIR)

@tool("local_image_search")
def local_image_search(query: str, top_k: int = 5):
    """
    Retrieve relevant local images using CLIP.
    Input: text query
    Output: list of image paths with similarity scores
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
    Fetch image URLs from Google Images.
    Returns a list of image URLs.
    """
    urls = fetch_google_images(query, num_images=num_images)

    return [{"url": u} for u in urls]

@tool("realtime_image_retrieval")
def realtime_image_retrieval(query: str, num_images: int = 20, top_k: int = 5):
    """
    Fetch images from the web, download them, and rerank using CLIP.
    Returns top local image paths with similarity scores.
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