# Local AI Platform Backend

Local-only FastAPI backend for Ollama. The service binds to `127.0.0.1` by default and does not use hosted inference APIs.

## Setup

From PowerShell in this directory:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The existing PyTorch/CUDA installation is intentionally not included in `requirements.txt` and should not be reinstalled.

## Start

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET http://127.0.0.1:8000/api/v1/health`
- `GET http://127.0.0.1:8000/api/v1/models`
- `POST http://127.0.0.1:8000/api/v1/chat`

Example chat request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/chat -ContentType 'application/json' -Body '{"model":"qwen2.5:3b","message":"Say hello in one sentence."}'
```

## Test

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

Tests mock runtime HTTP calls. A running Ollama instance is required for the real endpoint verification.

## Frontend

The React/Vite frontend lives in `frontend/` and calls only the local backend. The backend allows the two local Vite development origins by default.

From a second PowerShell window:

```powershell
cd D:\endurava\AI\local-ai-platform\frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/`. The frontend loads health and model data from `http://127.0.0.1:8000` and sends chat requests to the existing local API. For a production bundle, run `npm run build` in `frontend/`.

## Task-aware model registry

The curated catalog is stored in `app/registry/model_manifest.json`. It records exact Ollama identifiers, supported tasks and input modalities, approximate resource requirements, license status, official model-card URLs, verification date, and limitations. It does not download models.

`GET /api/v1/models` joins this catalog with Ollama's live installed-model list. A model is only marked `installed: true` when Ollama reports it; `runtime_status: installed_not_loaded` means it is installed but the application has not verified that it is currently loaded. `GET /api/v1/models?task=coding` filters the joined registry by task.

`GET /api/v1/tasks` returns preferred model identifiers and whether each task pipeline is implemented. Chat and coding use the existing local chat endpoint. OCR now has a document workflow; embeddings remain cataloged but unavailable until their dedicated workflow is implemented. The lifecycle policy prefers one generation model at a time and leaves loading/offloading to Ollama; it does not terminate processes or automatically download models.

## Stage 4A OCR workflow

Document endpoints:

- `POST /api/v1/documents/upload` with multipart field `file`
- `GET /api/v1/documents`
- `GET /api/v1/documents/{document_id}`
- `GET /api/v1/documents/{document_id}/text`
- `DELETE /api/v1/documents/{document_id}`

Documents are stored below `data/documents/` under generated IDs. Defaults are
25 MB per upload, 100 PDF pages, 300 DPI maximum rendering, and one OCR job at
a time. PyMuPDF 2.0 is installed in the backend `.venv` and provides native
digital-PDF extraction plus single-page rendering. Scanned pages and images
use the selected Tesseract or GLM-OCR provider. PaddleOCR PP-OCRv5 remains
deferred because its package environment is unavailable. The frontend OCR
workspace displays status, page-level output, engines, warnings, and copyable
text.

PaddleOCR installation is intentionally deferred. If its package environment
becomes available later, install only its isolated worker dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.ocr-venv\Scripts\python.exe -m pip install paddlepaddle==3.2.0 paddleocr
```

The Paddle worker reports GPU use only when PaddlePaddle reports CUDA support;
PyTorch CUDA availability is not treated as proof of Paddle GPU acceleration.

### Local OCR benchmark

The benchmark is optional and never runs as part of normal document processing.
Register a corpus in a versioned `manifest.json` containing `version: "1.0"`
and a `documents` list. Each document has a local relative `path`,
`document_id`, `document_type`, `language`, and `pages` with `page_number`.
Ground truth can be supplied inline as `expected_text` or through a UTF-8
page-level `reference_path`. Documents without ground truth remain eligible
for timing and reliability measurements but are excluded from accuracy scores.
Paths are constrained to the manifest directory; missing files are reported
rather than fabricated.

Generate repeatable local fixtures and run all available engines sequentially:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_ocr.py `
  --generate-synthetic data\ocr-benchmarks\synthetic `
  --output-dir data\ocr-benchmarks\runs\stage-4a3 `
  --engines native,tesseract,glm-ocr
```

The command writes `results.json` and `report.md`. Native PyMuPDF is measured
as digital extraction, not OCR. CER, WER, exact match, empty-output and
manual-review flags are included when ground truth exists. GLM-OCR is run
directly for controlled comparison; production fallback frequency is not
inferred from direct runs. GPU memory is sampled with `nvidia-smi` when
available; process-RAM peak remains unavailable because no extra package is
installed. List and validate a corpus without inference, run a quick subset,
or regenerate a Markdown report from saved JSON:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_ocr.py --manifest C:\path\to\manifest.json --list
.\.venv\Scripts\python.exe scripts\benchmark_ocr.py --manifest C:\path\to\manifest.json --validate
.\.venv\Scripts\python.exe scripts\benchmark_ocr.py --manifest C:\path\to\manifest.json --quick --ids doc-001 --engines tesseract
.\.venv\Scripts\python.exe scripts\benchmark_ocr.py --report-from data\ocr-benchmarks\runs\latest\results.json
```

A user corpus of approximately 50 documents can be evaluated with:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_ocr.py `
  --manifest C:\path\to\corpus\manifest.json `
  --output-dir data\ocr-benchmarks\runs\my-corpus `
  --engines native,tesseract,glm-ocr
```

### Optional Tesseract engine

Tesseract can be selected in the OCR workspace for PNG/JPEG uploads. The
default executable is `C:\Program Files\Tesseract-OCR\tesseract.exe`, the
default language is `eng` (the API's `en` value is mapped to Tesseract's
`eng` traineddata), and execution is CPU-only. Override `TESSERACT_PATH` or
`TESSERACT_TIMEOUT_SECONDS` in the environment if needed. The provider checks
`tesseract --list-langs` before every job and does not claim unsupported
languages. PDFs are not passed directly to Tesseract.

## Stage 4B local document Q&A

RAG uses the installed `nomic-embed-text:latest` model through Ollama's local
`/api/embed` endpoint. The actual embedding dimension is read from the runtime
and recorded; the verified local model returned 768 dimensions. Embeddings and
page-aware chunks are stored in `data/rag/index.sqlite3`. This is an exact
cosine-similarity store intended for modest local collections and can later be
replaced behind the `RagStore` interface.

Indexing accepts completed or partially completed documents with extracted
text. Chunks preserve document ID, filename, original page number, extraction
method, OCR engine, stable content hash, and embedding metadata. Re-indexing
replaces a document's chunks transactionally; unchanged documents are skipped.
Deleting a document removes its index entries. The default retrieval threshold
is `RAG_MIN_SIMILARITY=0.25`, and context is capped at 12,000 characters.
`RAG_STORAGE_PATH` and `RAG_MIN_SIMILARITY` can be overridden locally.

RAG endpoints:

- `POST /api/v1/rag/index` with `{"document_ids": ["..."]}`
- `GET /api/v1/rag/indexes`
- `GET /api/v1/rag/indexes/{document_id}`
- `DELETE /api/v1/rag/indexes/{document_id}`
- `POST /api/v1/rag/query` with `question`, installed chat `model`, optional
  `document_ids`, and bounded `top_k`

Answers use only retrieved evidence and are generated by the selected local
Ollama chat model. Citations are constructed from persisted chunk metadata;
model-generated filenames and page numbers are not trusted. If no passage meets
the retrieval threshold, the service returns a no-evidence response without
citations. Similarity is a retrieval signal, not proof of factual correctness.
Uploaded document text is treated as untrusted data, and embedded instructions
are explicitly excluded from system behavior.

Run the real local smoke test after Ollama has both models installed:

```powershell
.\.venv\Scripts\python.exe scripts\test_rag_local.py
```

It uses temporary local data, verifies the embedding dimension, indexes known
digital/OCR page text, queries `qwen2.5:3b`, and checks citations.

Newly uploaded documents are processed sequentially: OCR/text extraction
finishes first, then successful documents are automatically indexed with
`nomic-embed-text:latest`. Chunks and embeddings are persisted in the SQLite
RAG store, so unchanged documents are reused instead of re-embedded. The RAG
workspace refreshes index status automatically. Each answer reports total
time, retrieval/embedding time, and Ollama generation time.

## Stage 4B.1 RAG latency controls

RAG queries pass bounded Ollama generation options by default:
`RAG_NUM_PREDICT=1024` and `RAG_TEMPERATURE=0.2`. For Qwen3 only, the
`think` request field can explicitly enable or disable thinking. Other models
do not receive the `think` parameter. Direct Qwen3 reasoning output is not
shown to users; incomplete reasoning is rejected rather than displayed as an
answer. Granite is the default RAG model in the UI because it returns direct
grounded explanations reliably with the installed Ollama runtime. Set
`RAG_QWEN3_THINK=true` for deliberate Qwen3 reasoning when supported.

The API exposes Ollama timing metadata when supplied by the runtime: model
load, prompt evaluation, generation evaluation, token counts, tokens per
second, Ollama total time, plus end-to-end retrieval and generation timings.
Unavailable fields are returned as `null` and shown as unavailable in the UI.

Run the sequential local latency benchmark against an indexed document:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_rag_latency.py `
  --document-id YOUR_DOCUMENT_ID `
  --models qwen3:4b,granite4.1:3b,qwen2.5:3b `
  --output data\rag\latency-benchmark.json
```

The benchmark compares Qwen3's runtime default with `think: false`, runs other
models without the Qwen-specific parameter, records timing/token/citation
signals, and stores answer hashes rather than copying answer text into the
benchmark artifact. The first request may include a cold model load; warm/cold
state is not forced or inferred beyond Ollama's reported `load_duration`.
