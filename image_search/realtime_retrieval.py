import os
import argparse
import requests
from google_search import fetch_google_images
from clip_retrieval_local import index_local_images, retrieve_local

CACHE_DIR = "downloaded_images"
os.makedirs(CACHE_DIR, exist_ok=True)

def download_images(urls):
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