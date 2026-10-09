# Team Hydra: run the website on another Windows PC

This guide transfers the Docker image to another Windows x64 PC, then starts the frontend and API together. Run commands in PowerShell. Your assigned API key is intentionally absent from this guide and the image. Enter it privately on the destination PC.

## 1. Prepare Docker on the destination PC

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) and follow its WSL 2 setup instructions. Restart Windows if requested. Open Docker Desktop and wait for its engine to start. Use Linux containers.

```powershell
docker version
```

Continue only when this shows both Client and Server information. A missing `docker_engine` pipe means the engine is not ready. Start Docker Desktop and resolve any WSL or virtualization warning it displays.

Python, Node.js and CUDA are not required on the destination PC for the CPU image described here. Installation and image building need internet access; the transferred application image includes its runtime dependencies and bundled model.

## 2. Prepare the image on the original PC

Open PowerShell in the `tensorforge` folder containing `Dockerfile`, `app`, `web`, `specs` and `models`.

Choose ONE build below. The baseline is smaller; the neural build uses your trained model and runs inference on CPU.

### Option A: bundled baseline

```powershell
docker build -t team-hydra-api:transfer .
```

### Option B: your trained XLM-R model

Copy the entire selected model folder into `tensorforge/models/deployed`. It must contain `metadata.json`, `heads.safetensors`, and the complete `encoder` directory. Use your finalized model if available. The API ZIP does not include these large neural weights.

```powershell
docker build --build-arg MODEL_KIND=neural --build-arg DEFAULT_MODEL_PATH=/service/models/deployed -t team-hydra-api:transfer .
```

This image uses CPU PyTorch even if the PC has an RTX 4050. GPU inference requires a different CUDA image and GPU configuration; adding `--gpus all` alone is insufficient.

### Save the built image

Only continue after the build succeeds:

```powershell
docker image save --output team-hydra-api.tar team-hydra-api:transfer
Get-FileHash .\team-hydra-api.tar -Algorithm SHA256
```

Copy `team-hydra-api.tar` and this guide to the destination PC using a USB drive or private transfer. Record the hash to check the copy. Image save/load transfers the application image, not existing containers, API-key environment settings, external model mounts, or job volumes. By building with the model inside the image as above, no separate model mount is needed. Old jobs will not appear on the new PC.

Docker documents this transfer workflow in its [backup and restore guide](https://docs.docker.com/desktop/settings-and-maintenance/backup-and-restore/).

## 3. Load the image on the destination PC

Open PowerShell in the folder containing the transferred file:

```powershell
Get-FileHash .\team-hydra-api.tar -Algorithm SHA256
docker image load --input .\team-hydra-api.tar
docker image ls team-hydra-api
```

The checksum should match the original. The image list should show the `transfer` tag.

## 4. Enter the key privately and start the container

Run these commands in the SAME PowerShell window. Enter the exact assigned key at the hidden prompt, without quotation marks or extra spaces.

```powershell
$secret = Read-Host 'Enter Team Hydra API key' -AsSecureString
$env:API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
if ([string]::IsNullOrEmpty($env:API_KEY)) { throw 'API key must not be empty.' }

docker run -d --name team-hydra-api --restart unless-stopped -p 127.0.0.1:8000:8000 --env API_KEY --mount type=volume,source=hydra-jobs,target=/service/runtime team-hydra-api:transfer
```

This exposes the site only on the destination PC and preserves new jobs in the `hydra-jobs` volume. The key is passed from the environment, not written into command history. Docker stores it in the container configuration; people with Docker administrator access can inspect it. Do not share `docker inspect` output.

If Docker reports that the name already exists, run `docker ps -a` to identify it. If it is already the correct application, use `docker start team-hydra-api`. To replace this application's container or change its key, first finish any active jobs, then run:

```powershell
docker stop team-hydra-api
docker rm team-hydra-api
```

Then repeat the `docker run` command above with the key still set in that PowerShell window. The named volume remains. A restart alone does not update the container's key or image.

## 5. Verify the API and frontend

```powershell
docker ps --filter name=team-hydra-api
docker logs --tail 40 team-hydra-api
Invoke-RestMethod 'http://localhost:8000/health'
```

Wait for the startup-complete message, particularly for the neural model. Check the model version returned by `/health` to ensure you deployed the intended model.

With the key still set in this PowerShell window, test a prediction:

```powershell
$body = @{channel='chat'; text='I was charged twice. Please refund the duplicate payment.'} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8000/predict' -Method Post -Headers @{'X-API-Key'=$env:API_KEY} -ContentType 'application/json' -Body $body
```

A successful response contains the prediction. Then open **http://localhost:8000/** in a browser. Enter the same key and click **Use key**, then **Try an example** and **Classify ticket**. `/docs` is API documentation; `/` is the frontend.

After testing, clear the temporary key from PowerShell:

```powershell
Remove-Item Env:API_KEY -ErrorAction SilentlyContinue
$secret.Dispose()
```

The running container retains its configured key. The browser keeps its key only in tab memory and clears it on refresh.

## 6. Everyday commands

```powershell
# Stop the application
docker stop team-hydra-api

# Start it again with its existing key, model and volume
docker start team-hydra-api

# Read startup errors
docker logs --tail 60 team-hydra-api
```

After a Windows restart, Docker Desktop must be running for containers to start. Keep one application container using the jobs volume; this SQLite backend does not support multiple replicas. Do not delete `hydra-jobs` if you want to preserve its job records.

## Troubleshooting

| Symptom | Action |
|---|---|
| Cannot connect to Docker / missing pipe | Open Docker Desktop; wait for the engine; check `docker version`. |
| Port 8000 already allocated | Stop the old service you recognize, or recreate this container with `-p 127.0.0.1:8001:8000` and browse to port 8001. |
| Browser cannot connect | Check `docker ps -a` and container logs; wait for model loading to finish. |
| HTTP 401 | Ensure the browser/test uses the same exact key entered before container creation. Recreate the container to change the key. |
| API works but no frontend | Rebuild from the updated project with its `web` folder and `COPY web ./web` Dockerfile line; save, transfer and load the new image, then recreate the container. |
| Container exits during model loading | Check logs for missing model files or memory errors. Copy the full model directory and allow enough Docker/WSL memory. |
| Wrong model predictions/version | Confirm you used the intended build option; the standard build uses the baseline. |

## Local setup versus public hosting

This procedure runs the website on the other PC. It does not create a public URL. A public tunnel or hosting deployment is a separate step. Verify local authentication and predictions first. For judging, the chosen host, URL, model and exact assigned key must remain available throughout evaluation.

## Verification scope

These instructions match the supplied project's Dockerfile and API configuration. Docker image building and destination-PC execution could not be tested from the authoring environment because Docker was unavailable there.
