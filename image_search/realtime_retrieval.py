import os
import argparse
import requests
from google_search import fetch_google_images
from clip_retrieval_local import index_local_images, retrieve_local

CACHE_DIR = "downloaded_images"
os.makedirs(CACHE_DIR, exist_ok=True)

def download_images(urls):
    """
    Download images from URLs and save them to cache directory.
    
    Downloads images with a 10-second timeout. Failed downloads are
    silently skipped. Images are saved with sequential filenames.
    
    Args:
        urls: List of image URL strings to download.
    
    Returns:
        list: List of local file paths for successfully downloaded images.
    """
    saved = []
    for i, u in enumerate(urls):
        try:
            r = requests.get(u, timeout=10)
            r.raise_for_status()
            ext = u.split("?")[0].split(".")[-1]
            path = os.path.join(CACHE_DIR, f"img_{i}.{ext}")
            with open(path, "wb") as f:
                f.write(r.content)
            saved.append(path)
        except Exception:
            continue
    return saved

def search_and_retrieve(prompt, num_images=20, top_k=5):
    """
    Search Google Images, download them, and retrieve top matches using CLIP.
    
    Complete pipeline that:
    1. Searches Google Images for the prompt
    2. Downloads images to cache directory
    3. Indexes downloaded images with CLIP
    4. Retrieves top-k most similar images
    
    Args:
        prompt: Text query string for image search.
        num_images: Number of images to fetch from Google (default: 20).
        top_k: Number of top results to return after CLIP ranking (default: 5).
    
    Returns:
        list: List of tuples (image_path, similarity_score) for top-k matches.
    """
    urls = fetch_google_images(prompt, num_images)
    download_images(urls)
    emb, paths = index_local_images(CACHE_DIR)
    return retrieve_local(prompt, emb, paths, top_k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Google image search + CLIP-based local retrieval"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for search and retrieval"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=20,
        help="Number of images to fetch from Google"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return"
    )

    args = parser.parse_args()

    results = search_and_retrieve(
        args.prompt,
        num_images=args.num_images,
        top_k=args.top_k
    )

    for path, score in results:
        print(f"{score:.3f} — {path}")