"""Vision-Language inference CLI (backward-compatible shim).

The implementation now lives in :mod:`image_understanding.vlm` with lazy model
loading. This module is kept as a thin entry point so existing invocations
(``python vl_inference.py <image> [prompt]``) keep working without loading the
model at import time.
"""

from image_understanding.vlm import describe_image


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vl_inference.py <image_path> [prompt]")
        raise SystemExit(1)
    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."
    print(describe_image(image_path, prompt))


if __name__ == "__main__":
    main()
