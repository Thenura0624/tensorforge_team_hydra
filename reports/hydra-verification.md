# Team Hydra API verification

## Completed checks

- Seven baseline/contract/authentication test methods passed in 12.353 seconds.
- The real 5,000-ticket baseline integration flow completed in 2.82 seconds.
- Concurrent in-process test: 120 requests at concurrency 12 during an active bulk job, with 80 expected 200 responses and 40 expected 401 responses. Both supported credential headers were tested.
- Twenty-four simultaneous submissions with one idempotency key returned one job ID.
- Exact-key matching, fail-closed configuration and startup-only key loading passed. Responses did not contain the disposable test key.
- Real HTTP load check: 128 requests at concurrency 16, with 96 expected 200 responses and 32 expected 401 responses, completed in 0.686 seconds during bulk inference. No unexpected status occurred.
- Real HTTP forced-process-restart recovery passed: the interrupted job persisted and returned failed/interrupted.
- The actual saved XLM-R artifact loaded offline on CPU. Eight predictions matched the original tournament loader, including labels and confidence (within numerical tolerance), and passed the supplied response schema.
- A separate XLM-R API integration test passed in 23.514 seconds, exercising health, single prediction, synchronous batch, asynchronous submission/status/results, deletion and expiry-after-discard.
- PowerShell private-key launcher syntax validated.

The tested neural version was `hydra-xlmr-f0899a3b764b3324`. These are software/inference checks, not a new accuracy evaluation or final-model confirmation.

## HTTP timings, bundled baseline

| Measurement | Development-host result |
|---|---:|
| Ready after startup | 5.77 s |
| Single prediction p95, 30 requests | 9.91 ms |
| 100-ticket batch p95, 10 requests | 31.49 ms |
| Accept 5,000 longer tickets | 686 ms |
| Health during bulk inference | 1.25 ms |
| Single prediction during bulk inference | 8.71 ms |

Raw results are in `http-smoke.json`. Traffic used synthetic/repeated examples and disposable test keys. The actual assigned evaluation key was not used or persisted. No reference-hardware or sustained endurance benchmark is claimed.

## Deployment limits

The SQLite service supports one process and one replica; use its persistent volume for container replacement. Docker build/run and Linux locking have not been verified on this Windows host. No public HTTPS hosting was performed. Neural model serving was functionally verified on CPU, but neural latency under the competition's 2-vCPU/4-GB reference hardware remains unverified.
