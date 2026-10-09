# Team Hydra — API handoff

All seven operations in the supplied v2.1.0 contract are implemented:

| Method | Endpoint | Limit / response |
|---|---|---|
| GET | `/health` | Public readiness and model version |
| POST | `/predict` | One ticket; 1 MiB body |
| POST | `/predict/batch` | 1–100 tickets; 5 MiB body |
| POST | `/batch/jobs` | 1–5,000 tickets; 25 MiB body; 202 + Location + Retry-After |
| GET | `/batch/jobs/{job_id}` | State and progress |
| GET | `/batch/jobs/{job_id}/results` | Ordered, paginated predictions |
| DELETE | `/batch/jobs/{job_id}` | Cancel active or discard finished job; 204 |

## Start privately on Windows

Open PowerShell in this API folder (`tensorforge`), then install dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\Start-Hydra.ps1
```

The launcher prompts for the assigned key with hidden input if `API_KEY` is not already supplied. Enter the **exact assigned key**, without the surrounding quotation marks used in an email or document. The launcher does not write it to a file or print it. Use your real evaluation key locally; the placeholder from the chat has not been embedded in this project.

The server listens on port 8000. `/docs` displays the contract. The key must be sent in either `X-API-Key` or `Authorization: Bearer`. It is never accepted through query strings.

## Web frontend

After starting the API, open http://localhost:8000 in your browser. The frontend is served by FastAPI and included in the Docker image; no separate frontend server or Node.js installation is required. Rebuild your image and recreate the container to include this update.

Enter the same API key configured on the server, then select **Use key**. The key stays only in tab memory and is cleared on refresh; it is not saved to browser storage. Use HTTPS when hosting remotely.

- **Single ticket:** enter a message and channel to see category, assigned team, urgency, and confidence.
- **Batch tickets:** upload JSON/JSONL or paste ticket data. Up to 100 tickets run immediately; 101–5,000 create a background job.
- **Job tracking:** check progress using the job ID, cancel active jobs, page through results, and download predictions as JSON. Save the job ID to return after refreshing.

The frontend uses whichever model the API has loaded. Its footer shows the model version.

## Authentication details

- `API_KEY` is read once when the server starts. The running process stores its SHA-256 digest for constant-time digest comparison.
- The exact string is preserved; the implementation does not trim it, rotate it, expire it, decrement an allowance or disable it after failed requests.
- Missing or incorrect credentials return JSON 401 with `WWW-Authenticate: Bearer`, before body parsing or schema validation.
- If the key is absent from configuration, protected routes fail closed with 401. The launcher refuses an empty key.
- Every job route is protected, including status, results and deletion. `/health` stays public.
- No request-header/body logging is added. The launcher/container disable Uvicorn access logs, and responses use `Cache-Control: no-store`.
- Both correct and incorrect credentials were exercised concurrently during bulk work; the tests use disposable test keys, not your assigned key.

For hosted deployment, put the exact key in the provider's private environment/secret setting named `API_KEY`. Keep that value unchanged across restarts throughout evaluation. Do not put it in the Dockerfile, a public repository, screenshots or command-line arguments. Your hosting provider must keep the service and its HTTPS URL available throughout the evaluation window.

## Use your selected XLM-R model

The existing tournament comparison identifies XLM-R as its internal-comparison winner. The API now supports the saved XLM-R/mDeBERTa encoder + heads directories, as well as the original baseline and SVM joblib bundles. A final full-training artifact is preferable when available; the comparison winner is not itself proof of final validation performance.

For a CPU API environment, install the optional neural dependencies:

```powershell
.\.venv\Scripts\python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements-neural.txt
.\Start-Hydra.ps1 -ModelPath ..\model-tournament\results_xlmr\model
```

If you have completed the tournament's `finalize` command, point `-ModelPath` to `..\model-tournament\best_model\model` instead. The model directory must contain `metadata.json`, `heads.safetensors`, and the `encoder` directory. SVM directories containing `metadata.json` and `model.joblib` also work.

Neural inference defaults to CPU with two threads. To use the RTX 4050, install the CUDA-enabled torch wheel in the API environment and set `$env:MODEL_DEVICE = 'cuda'` before launch. `MODEL_BATCH_SIZE` defaults to 2. Only one neural microbatch runs at a time to bound GPU memory; the lock is released between microbatches so live requests can interleave with bulk processing. All neural assets load from local files without runtime downloads.

The bundled default remains the original evaluated baseline. The large trained neural weights are not duplicated inside the API ZIP; select your existing local artifact as shown above. Inference fidelity was checked against the tournament loader, independently of accuracy evaluation.

## Container and persistent jobs

For the bundled baseline:

```powershell
docker build -t team-hydra-api .
docker run -p 8000:8000 --env API_KEY -v hydra-jobs:/service/runtime team-hydra-api
```

Here `--env API_KEY` takes an already configured environment value without putting its contents in the command. The supplied Compose file provides the same persistent volume and `restart: unless-stopped`:

```powershell
docker compose up --build -d
```

For an offline image containing the neural winner, first copy its **entire** saved model directory into `models/deployed` inside the build context, then build:

```powershell
docker build --build-arg MODEL_KIND=neural --build-arg DEFAULT_MODEL_PATH=/service/models/deployed -t team-hydra-api .
```

The resulting image starts with the same `docker run -p 8000:8000 -e API_KEY=<key> <image>` interface required by the organizers. Dependencies and model weights are included at build time; runtime has no download requirement. The Docker installation/build is not verified on this host because Docker is unavailable.

Run **one worker and one replica** with this SQLite backend. A process lock prevents competing workers from corrupting recovery state. One running job and three queued jobs are accepted; additional jobs receive 429. Finished jobs persist for at least six hours, expired IDs return 410, and a job interrupted by process restart becomes failed with code `interrupted`. Use the named volume across replacement containers. Scaling across replicas requires a distributed job store/worker design and is not supported by this package.

## Verify

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python tests/smoke_server.py
```

The suite covers supplied JSON Schemas, authentication ordering, Unicode input, payload limits, duplicate IDs, ordered predictions, a 5,000-ticket job, cancellation, retention, restart recovery, concurrent authentication and concurrent idempotent submissions. Test keys are randomly generated or clearly identified local test values; the assigned evaluation key is not needed for these tests.

See [current verification results](reports/hydra-verification.md). To enable the optional neural integration test, set `NEURAL_TEST_MODEL` to a saved model directory before running the test suite; it is skipped by default when neural weights are not supplied.

No public endpoint has been deployed by this task. Container deployment and reference-hardware load testing remain separate from local implementation verification.
