# Python files overview (concise)

This is a compact index of the main Python files in this repo.

- The `NeMo/` subtree is large vendored code and is intentionally not enumerated here.
- Many modules have deeper docs in `documentation/` (e.g., `router.md`, `ingest.md`, `llm.md`, etc.).

## Entrypoints (top-level)

- `cli.py`: Minimal interactive CLI that routes queries (RAG/META/CHAT) and streams responses; optionally prints confidence bars.
- `run.py`: Simple argparse entrypoint to run `--ingest` or ask a one-off question via `--ask`.
- `vl_inference.py`: Standalone vision-language inference script for Qwen2.5-VL (loads model/processor, prepares an image+text prompt, runs generation).

## Core package (`cortex/`)

- `cortex/embeddings.py`: Lazily creates and caches the HuggingFace embedding model (`all-MiniLM-L6-v2`) for vector search.
- `cortex/ingest.py`: Loads documents from `data/documents/`, chunks them, embeds them, and persists a Chroma vector DB in `chroma_db/`.
- `cortex/llm.py`: Returns an LLM backend: `LlamaCpp` on GPU if available, otherwise a lightweight HF text-generation pipeline fallback.
- `cortex/logo.py`: ASCII logo assets and Rich helpers to render styled/colored logo variants.
- `cortex/memory.py`: Simple in-process conversation memory (stores recent queries/answers with timestamps).
- `cortex/persona.py`: Defines the system prompt/persona string (`CORTEX_SYSTEM_PROMPT`).
- `cortex/query.py`: RAG + chat execution: builds/returns a Chroma retriever, formats context, runs RAG, META, or plain CHAT flows.
- `cortex/router.py`: TF‑IDF + LogisticRegression router that picks `rag`/`meta`/`chat` and executes the chosen handler with confidence scores.
- `cortex/streaming.py`: LangChain callback handler used to stream tokens to a caller-provided function.
- `cortex/ui.py`: Rich + prompt_toolkit interactive terminal UI with streaming, commands (`ingest`, `status`, `history`, etc.), and basic source display.

## Reinforcement-learning router tools (`cortex/rl_*`)

- `cortex/rl_router.py`: RL-style wrapper around the base TF‑IDF router; records feedback, updates per-route confidence weights, persists metrics/feedback.
- `cortex/rl_cli.py`: Interactive “training” CLI: shows predicted route, asks user for the correct category, learns from feedback, then executes; optional TTS.
- `cortex/rl_dashboard.py`: ASCII dashboard for RL training progress (learning curve, per-route accuracy, confusion matrix, weight adjustments, recent mistakes).

## Image search (`image_search/`)

- `image_search/google_search.py`: Google Images search helper (requires env vars `GOOGLE_API_KEY` and `CX`) returning image URLs.
- `image_search/realtime_retrieval.py`: End-to-end pipeline: Google image search → download into cache → CLIP index locally → retrieve top matches for a prompt.
- `image_search/clip_retrieval_local.py`: Local CLIP embedding + retrieval over images in a directory (indexes image features, ranks by text/image similarity).

## Speech to text (`speech_to_text/`)

- `speech_to_text/audio_recorder.py`: Simple terminal audio recorder using `sounddevice`; saves a timestamped `.wav`.
- `speech_to_text/parakeet_asr.py`: Thin wrapper around a NeMo ASR model (“Parakeet”) to transcribe a wav file.
- `speech_to_text/record_and_transcribe.py`: Glue script: record audio with `AudioRecorder` then transcribe with `ParakeetASR`.

## Text to speech (`text_to_speech/`)

- `text_to_speech/text_to_speech.py`: Wrapper around Coqui TTS to synthesize speech to a wav and optionally play it via `sounddevice`.
- `text_to_speech/testing.py`: Small quick-start script to synthesize a sample line to `logs/output.wav`.

## TF‑IDF classifier (`tf-idf_classifier/`)

- `tf-idf_classifier/train_classifier.py`: Trains TF‑IDF + LogisticRegression router classifier from labeled text files; logs metrics to W&B; saves `joblib` artifacts.
- `tf-idf_classifier/inference.py`: Loads the saved vectorizer/classifier and predicts label + confidence scores for single/batch texts.

## Utilities (`utils/`)

- `utils/data_cleaner.py`: One-off script to clean `dataset/meta.txt` (remove numbering/separators) and write a simplified `meta.txt`.

