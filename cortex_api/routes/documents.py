"""Documents route — list the corpus and ingest uploads into the vector store."""

import os
import time
from datetime import UTC, datetime

from fastapi import APIRouter, File, HTTPException, UploadFile

from cortex.config import DATA_DIR

router = APIRouter(prefix="/api", tags=["Documents"])

# Extensions the ingest pipeline (cortex.ingest.load_file) can actually load.
SUPPORTED_EXTS = {".pdf", ".txt", ".docx"}

_TYPE_LABELS = {
    ".pdf": "PDF",
    ".txt": "Text",
    ".docx": "Word",
    ".md": "Markdown",
}


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.0f} GB"


def _human_age(mtime: float) -> str:
    delta = max(0, int(time.time() - mtime))
    if delta < 60:
        return "just now"
    for secs, label in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if delta >= secs:
            n = delta // secs
            return f"{n} {label}{'s' if n != 1 else ''} ago"
    return "just now"


@router.get("/documents")
def list_documents() -> dict:
    """List the supported documents currently in ``DATA_DIR``."""
    documents = []
    if os.path.isdir(DATA_DIR):
        for name in sorted(os.listdir(DATA_DIR)):
            path = os.path.join(DATA_DIR, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in _TYPE_LABELS:
                continue
            st = os.stat(path)
            documents.append(
                {
                    "name": name,
                    "type": _TYPE_LABELS[ext],
                    "size": _human_size(st.st_size),
                    "updated": _human_age(st.st_mtime),
                    # Raw values so the UI can sort by size and modification time.
                    "sizeBytes": st.st_size,
                    "modifiedAt": datetime.fromtimestamp(st.st_mtime, UTC).isoformat(),
                }
            )
    return {"documents": documents, "indexedCount": len(documents)}


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Save an uploaded file to ``DATA_DIR`` and ingest it into the vector store."""
    filename = os.path.basename(file.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTS))}.",
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, filename)
    contents = await file.read()
    with open(dest, "wb") as fh:
        fh.write(contents)

    # Lazy import keeps heavy ingest deps out of app startup.
    from cortex.ingest import ingest_file

    try:
        chunks_added = ingest_file(dest)
    except Exception as exc:  # noqa: BLE001 — surface ingest failures to the client
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from None

    return {
        "name": filename,
        "chunksAdded": chunks_added,
        "uploadedAt": datetime.now(UTC).isoformat(),
    }
