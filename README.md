# Team Hydra — TensorForge support classification API

See [Team Hydra setup and key handling](TEAM_HYDRA.md) for private API-key entry, the complete endpoint list, concurrent-authentication verification, and loading the selected XLM-R/SVM artifact.

Trained multilingual ML model and all seven API operations from the supplied v2.1.0 contract. Inference is local and requires no internet or external LLM. The exact supplied contract and JSON Schemas are in `specs/`.

## Run locally

The integrated Team Hydra frontend is available at after starting the API. It supports single tickets, JSON/JSONL batches, background jobs, and result downloads. Enter your configured API key in the page; it is held only in tab memory. See `TEAM_HYDRA.md` for the private-key launcher.

Use Python 3.14 (tested with 3.14.3). From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:API_KEY = 'replace-with-your-private-key'
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

On Linux/macOS, activate the virtual environment and set `export API_KEY='your-private-key'` before the same Uvicorn command. Use the installed Python interpreter if `python` resolves to another environment. Do not commit your real API key.

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"ticket_id":"demo-1","channel":"chat","text":"I was charged twice for my ride."}'
```

Interactive API documentation is at `/docs`. Both `X-API-Key` and `Authorization: Bearer` are accepted. The configured key is read at startup; missing configuration fails closed with 401.

| Operation | Behavior |
|---|---|
| `GET /health` | Public readiness and artifact-derived model version |
| `POST /predict` | One ticket, optional ticket ID |
| `POST /predict/batch` | 1–100 tickets, required unique IDs, ordered results |
| `POST /batch/jobs` | 1–5,000 tickets; 202, Location and Retry-After headers |
| `GET /batch/jobs/{job_id}` | Status and monotonic progress |
| `GET /batch/jobs/{job_id}/results` | Ordered results; offset ≥ 0, limit 1–5,000 |
| `DELETE /batch/jobs/{job_id}` | Cancel active jobs or discard finished results |

Bodies are checked in order: authentication, content type, streamed size limit, JSON parsing, schema validation. Limits are 1/5/25 MiB respectively for single, synchronous batch and job submission. Unknown fields are accepted as the schemas specify, but never used as model features. Error responses are JSON. Request IDs are echoed.

## Persistent asynchronous jobs

SQLite stores tickets, progress, results and idempotency keys. One dedicated background thread processes chunks of 100, leaving request handling available. Capacity is four outstanding jobs: one running plus three waiting. Finished jobs are retained for at least six hours; expired IDs retain a small tombstone and return 410. Unknown IDs return 404. Expiry cleanup runs in the worker loop. The same idempotency key returns the original job; after expiry it returns 410, so use a new key for a new run.

**Supported deployment: one Uvicorn process, one replica.** A database process lock rejects a second process using the same DB. Do not scale this configuration across replicas or independent databases. SQLite state survives process restarts; queued/running jobs become failed with `error.code=interrupted`. Run the container with a persistent volume to preserve this behavior when replacing a container. For multi-replica hosting, replace this job runner with a shared database/queue and distributed worker leases.

Environment overrides: `MODEL_PATH` selects a trusted local model; `JOB_DB_PATH` selects the SQLite path. Never load untrusted joblib files.

## Docker

```bash
docker build -t tensorforge-baseline .
docker run -p 8000:8000 -e API_KEY="$API_KEY" tensorforge-baseline
```

For durable state across replacement containers:

```bash
docker run -p 8000:8000 -e API_KEY="$API_KEY" \
  -v tensorforge-jobs:/service/runtime tensorforge-baseline
```

Dependencies are installed during the image build; the trained model is bundled. Runtime requires no downloads. The image runs as a non-root user. Docker is unavailable on the development host, so the image build and container launch have **not** been verified here. No public deployment was performed.

## Reproduce training and evaluation

```bash
python train.py
python -m unittest discover -s tests -v
python tests/smoke_server.py
```

Training uses only the 4,000 supplied training rows. The 800 validation rows are evaluation-only; no hyperparameters were selected on them. The included JSONL files are the original UTF-8 datasets. CSV duplicates were not needed. The feature pipeline uses channel, subject and body, not ticket ID or language. Word 1–2 grams and character 3–5 grams feed three logistic-regression classifiers; urgency uses balanced class weights. Primary probabilities supply confidence, without post-hoc calibration. Deterministic consistency rules enforce category/team mapping, distinct secondary labels, and nonurgent/no-secondary spam.

`reports/metrics.json` includes per-class precision/recall/F1, language breakdowns, confusion matrix, joint accuracy, log loss, data hashes and artifact version. `reports/validation_predictions.jsonl` contains every validation prediction. See `reports/evaluation.md` for interpretation. Retraining creates a new artifact hash and therefore may change the model version; keep the included artifact for the reported results.

All five integration test methods passed, including a 5,000-ticket job. A real HTTP smoke test also passed, including forced-process-restart recovery. See `reports/verification.md` for measured timings and remaining verification limits.

The pasted planning document was treated as context. The supplied OpenAPI/JSON Schemas control implementation where they differ: DELETE is required, asynchronous jobs accept 1–5,000 items, and language is not an API feature. Dashboard creation, hosting, competition submission and video production are outside this implementation.
