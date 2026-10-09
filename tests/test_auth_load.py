"""Auth correctness under concurrent inference and asynchronous-job traffic."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import secrets
import tempfile
import threading
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.auth import APIKeyAuth
from app.main import create_app

class AuthenticationLoadTests(unittest.TestCase):
    def test_exact_key_and_fail_closed(self):
        assigned = ' exact local test key '
        auth = APIKeyAuth(assigned)
        self.assertTrue(auth.accepts({'x-api-key': assigned}))
        self.assertTrue(auth.accepts({'authorization': 'Bearer ' + assigned}))
        self.assertFalse(auth.accepts({'x-api-key': assigned.strip()}))
        self.assertFalse(APIKeyAuth(None).accepts({'x-api-key': ''}))

    def test_concurrent_auth_and_idempotency_during_bulk(self):
        assigned = secrets.token_urlsafe(32)
        ticket = dict(ticket_id='load-ticket', channel='chat', text='Please refund my duplicate payment.')
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'API_KEY': assigned}):
            app = create_app(db_path=Path(temporary) / 'jobs.sqlite3')
            with TestClient(app) as client:
                # A later environment change must not rotate the running server's key.
                os.environ['API_KEY'] = 'changed-after-startup-test-value'
                original = app.state.jobs.model
                entered, release = threading.Event(), threading.Event()
                class SlowBulk:
                    version = original.version
                    def predict(self, tickets):
                        entered.set()
                        if not release.wait(20):
                            raise RuntimeError('Test worker gate timed out')
                        return original.predict(tickets)
                app.state.jobs.model = SlowBulk()
                try:
                    accepted = client.post('/batch/jobs', json={'tickets': [ticket]}, headers={'X-API-Key': assigned})
                    self.assertEqual(accepted.status_code, 202)
                    self.assertTrue(entered.wait(5))
                    def request(index):
                        kind = index % 6
                        if kind == 0:
                            response = client.post('/predict', json=ticket, headers={'X-API-Key': assigned})
                            expected = 200
                        elif kind == 1:
                            response = client.post('/predict', json=ticket, headers={'Authorization': 'Bearer ' + assigned})
                            expected = 200
                        elif kind == 2:
                            response = client.post('/predict', content='invalid-json', headers={'X-API-Key': 'wrong', 'Content-Type': 'application/json'})
                            expected = 401
                        elif kind == 3:
                            response = client.post('/batch/jobs', content='invalid-json')
                            expected = 401
                        elif kind == 4:
                            response = client.get('/health')
                            expected = 200
                        else:
                            response = client.get(accepted.headers['location'], headers={'Authorization': 'Bearer ' + assigned})
                            expected = 200
                        self.assertEqual(response.status_code, expected, response.text)
                        self.assertNotIn(assigned, response.text)
                        self.assertEqual(response.headers['cache-control'], 'no-store')
                        if expected == 401:
                            self.assertEqual(response.headers['www-authenticate'], 'Bearer')
                        return response.status_code
                    with ThreadPoolExecutor(max_workers=12) as pool:
                        statuses = list(pool.map(request, range(120)))
                    self.assertEqual(statuses.count(200), 80)
                    self.assertEqual(statuses.count(401), 40)
                    def retry(_):
                        response = client.post('/batch/jobs', json={'tickets': [ticket]},
                                               headers={'X-API-Key': assigned, 'Idempotency-Key': 'concurrent-retry'})
                        self.assertEqual(response.status_code, 202, response.text)
                        return response.json()['job_id']
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        ids = list(pool.map(retry, range(24)))
                    self.assertEqual(len(set(ids)), 1)
                    self.assertEqual(client.get('/batch/jobs/unknown/results', headers={'X-API-Key': 'wrong'}).status_code, 401)
                    self.assertEqual(client.delete('/batch/jobs/unknown', headers={'X-API-Key': 'wrong'}).status_code, 401)
                finally:
                    release.set()

if __name__ == '__main__':
    unittest.main(verbosity=2)
