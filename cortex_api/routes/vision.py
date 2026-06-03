"""Vision route — image + text Q&A via Qwen2.5-VL."""

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/api", tags=["Vision"])


@router.post("/vlm")
async def describe(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image."),
) -> dict:
    """Describe / answer a question about an uploaded image."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    suffix = os.path.splitext(image.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        img_path = tmp.name

    from image_understanding.vlm import describe_image

    try:
        description = describe_image(img_path, prompt)
    except Exception as exc:  # noqa: BLE001 — surface inference failures
        raise HTTPException(status_code=500, detail=f"Vision inference failed: {exc}") from None
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

    return {"description": description}
