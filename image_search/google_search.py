import os
import sys
import argparse
from dotenv import load_dotenv, find_dotenv
from google_images_search import GoogleImagesSearch

sys.path.append('../..')
_ = load_dotenv(find_dotenv())

API_KEY = os.environ["GOOGLE_API_KEY"]
CX = os.environ["CX"]

gis = GoogleImagesSearch(API_KEY, CX)

def fetch_google_images(prompt, num_images=10):
    """
    Return a list of image URLs for the given prompt.
    """
    search_params = {
        "q": prompt,
        "num": num_images,
        "fileType": "jpg|png",
        "safe": "active",
    }

    gis.search(search_params=search_params)
    return [img.url for img in gis.results()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch image URLs from Google Images")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Search prompt"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to fetch"
    )

    args = parser.parse_args()

    urls = fetch_google_images(args.prompt, num_images=args.num_images)
    for url in urls:
        print(url)