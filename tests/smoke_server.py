"""Real HTTP startup, latency and forced-process-restart checks."""
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import httpx
import numpy as np
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory() as temporary:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = {**os.environ, 'API_KEY': secrets.token_urlsafe(32),
               'JOB_DB_PATH': str(Path(temporary) / 'jobs.sqlite3'),
               'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
        client = httpx.Client(base_url=f'http://127.0.0.1:{port}', headers={'X-API-Key': env['API_KEY']}, timeout=30)
        report = {}
        process = None
        def launch():
            started = time.perf_counter()
            proc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1',
                                     '--port', str(port), '--workers', '1', '--no-access-log'],
                                    cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError('Server exited during startup')
                try:
                    if client.get('/health').status_code == 200:
                        return proc, time.perf_counter() - started
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            proc.kill()
            proc.wait()
            raise RuntimeError('Server did not become ready')
        try:
            process, report['startup_seconds'] = launch()
            ticket = dict(ticket_id='smoke', channel='chat', text='My food is late and I need a refund.')
            for path, payload, count in [('/predict', ticket, 30), ('/predict/batch', {'tickets': [
                    {**ticket, 'ticket_id': str(i)} for i in range(100)]}, 10)]:
                timings = []
                for _ in range(count):
                    started = time.perf_counter()
                    response = client.post(path, json=payload)
                    response.raise_for_status()
                    timings.append(time.perf_counter() - started)
                report[path] = {'requests': count, 'p95_seconds': float(np.percentile(timings, 95))}
            long_tickets = [{**ticket, 'ticket_id': str(i), 'text': ticket['text'] * 90} for i in range(5000)]
            started = time.perf_counter()
            job = client.post('/batch/jobs', json={'tickets': long_tickets})
            job.raise_for_status()
            report['5000_ticket_submission_seconds'] = time.perf_counter() - started
            location = job.headers['location']
            self_state = client.get(location).json()
            assert self_state['status'] in ('queued', 'running'), self_state
            started = time.perf_counter()
            assert client.get('/health').status_code == 200
            report['health_during_job_seconds'] = time.perf_counter() - started
            started = time.perf_counter()
            assert client.post('/predict', json=ticket).status_code == 200
            report['predict_during_job_seconds'] = time.perf_counter() - started
            def concurrent_request(index):
                kind = index % 4
                if kind == 0:
                    r = client.post('/predict', json=ticket)
                elif kind == 1:
                    r = client.post('/predict', json=ticket, headers={'X-API-Key': '', 'Authorization': 'Bearer ' + env['API_KEY']})
                elif kind == 2:
                    r = client.post('/predict', content='malformed', headers={'X-API-Key': 'incorrect-test-key'})
                else:
                    r = client.get('/health')
                assert r.status_code == (401 if kind == 2 else 200), r.status_code
                assert env['API_KEY'] not in r.text
                return r.status_code
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=16) as pool:
                statuses = list(pool.map(concurrent_request, range(128)))
            report['concurrent_auth'] = dict(requests=128, concurrency=16, successful_200=statuses.count(200),
                                             expected_401=statuses.count(401), seconds=time.perf_counter() - started)
            process.kill()
            process.wait(timeout=10)
            process, report['restart_seconds'] = launch()
            state = client.get(location).json()
            assert state['status'] == 'failed', state
            assert state['error']['code'] == 'interrupted', state
            report['forced_restart_recovery'] = 'passed: persisted job failed with interrupted'
            report['note'] = 'Measured on development host, not the competition reference hardware. Docker not tested.'
            (ROOT / 'reports' / 'http-smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report, indent=2))
        finally:
            client.close()
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=15)

if __name__ == '__main__':
    main()
