# Verification performed

Tests ran on 9 October 2026 with Python 3.14.3 and the versions pinned in `requirements.txt`.

## Integration suite

Command: `python -m unittest discover -s tests -v`

Result: **5 test methods passed**, 5.275 seconds total. These methods exercise multiple scenarios:

- Both authentication headers, missing key, fail-closed configuration and authentication before parsing.
- Public health, model version agreement, request-ID echo, Sinhala/Tamil/transliterated/emoji input, ordered batch results, and identical single/batch outputs.
- Invalid JSON, nonfinite JSON constants, deeply nested malformed JSON, missing/wrong fields, unknown channels, whitespace-only text, length limits, missing/duplicate IDs, invalid batch sizes, payload byte limits, unknown routes and unsupported methods.
- A 5,000-ticket job, idempotency, status/progress, pagination, result ordering, result/schema validation and explicit discard. Full test flow completed in 2.71 seconds.
- One active plus three queued jobs, queue overflow, not-ready results, cancellation, six-hour retention, expired IDs and persisted interruption recovery.

Response bodies were validated against the **supplied** JSON Schemas. Temporary job databases were isolated from normal runtime state.

## Real HTTP server

Command: `python tests/smoke_server.py`

| Check | Observed |
|---|---:|
| Startup to HTTP health ready | 4.34 s |
| Single prediction p95, 30 requests | 6.09 ms |
| 100-ticket batch p95, 10 requests | 19.42 ms |
| Submit 5,000 longer tickets | 358.26 ms |
| Health during active bulk inference | 0.92 ms |
| Single prediction during active bulk inference | 6.64 ms |
| Startup after forced process termination | 4.30 s |

The process was killed during an unfinished job and restarted with the same SQLite database. The job persisted and returned `failed` with `error.code=interrupted` as required. Raw measurements are in `http-smoke.json`; latency samples used repeated synthetic tickets, not a diverse load test. These are development-host measurements, not certification on the competition's 2-vCPU/4-GB reference hardware.

The initial test attempt encountered sandbox restrictions on local sockets and the default temporary directory. Network permission and a workspace-local temporary directory resolved those environment issues; no application socket workaround was added.

## Not verified

Docker build/start: Docker is not installed on this host. Linux-specific file locking is implemented but not exercised here. Public HTTPS hosting and multi-replica operation were not tested; this version explicitly supports a single process and replica. No competition submission was made.
