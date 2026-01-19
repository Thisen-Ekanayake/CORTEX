import os
import argparse
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# LOAD MODEL ONCE
model = CLIPModel.from_pretrained(
    "/ml/CORTEX/models/CLIP-vit-large-patch14",
    local_files_only=True
)
processor = CLIPProcessor.from_pretrained(
    "/ml/CORTEX/models/CLIP-vit-large-patch14",
    local_files_only=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def index_local_images(dir_path):
    paths = []
    embeddings = []

    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)

        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        paths.append(fpath)
        embeddings.append(emb.cpu())

    if embeddings:
        return torch.cat(embeddings, dim=0), paths

    return None, []

def retrieve_local(prompt, image_embeddings, image_paths, top_k=5):
    text_inputs = processor(
        text=[prompt],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)

    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

    sims = (image_embeddings.to(device) @ text_emb.T).squeeze(1)
    top_scores, top_idxs = sims.topk(top_k)

    return [(image_paths[i], float(top_scores[j])) for j, i in enumerate(top_idxs)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP local image retrieval")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image retrieval"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return"
    )

    args = parser.parse_args()

    image_embeddings, image_paths = index_local_images(args.images_dir)
    if image_embeddings is None:
        raise RuntimeError("No valid images found.")

    results = retrieve_local(
        args.prompt,
        image_embeddings,
        image_paths,
        top_k=args.top_k
    )

    for path, score in results:
        print(f"{score:.3f} — {path}")