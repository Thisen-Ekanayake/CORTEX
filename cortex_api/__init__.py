"""CORTEX AI API — FastAPI service exposing the cortex core to the web UI.

Launch from the repository root so that ``cortex.config`` root-relative paths
(``models/``, ``chroma_db/``, ``tf-idf_classifier/model/``) resolve correctly::

    uvicorn cortex_api.main:app --port 8001 --reload
"""
