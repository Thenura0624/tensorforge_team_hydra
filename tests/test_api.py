import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from app.main import create_app, ROOT
from app.jobs import Jobs, JobError
from app.auth import APIKeyAuth

KEY = 'local-test-only'
HEADERS = {'X-API-Key': KEY}
TICKET = dict(ticket_id='test-1', channel='chat', text='I was charged twice. Please refund the duplicate payment.')

def validate(name, payload):
    schema = json.loads((ROOT / 'specs' / (name + '.schema.json')).read_text(encoding='utf-8'))
    Draft202012Validator(schema).validate(payload)

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {'API_KEY': KEY})
        cls.env.start()
        cls.app = create_app(db_path=Path(cls.temp.name) / 'jobs.db')
        cls.client = TestClient(cls.app).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.env.stop()
        cls.temp.cleanup()

    def check_error(self, response, status):
        self.assertEqual(response.status_code, status, response.text)
        validate('error_response', response.json())

    def test_frontend_and_assets(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Team Hydra', response.text)
        self.assertIn("default-src 'self'", response.headers['content-security-policy'])
        for asset in ['app.js', 'styles.css', 'favicon.svg']:
            self.assertEqual(self.client.get('/ui/' + asset).status_code, 200)
        self.check_error(self.client.post('/predict', json=TICKET), 401)

    def test_auth_before_parsing(self):
        for path in ['/predict', '/predict/batch', '/batch/jobs']:
            r = self.client.post(path, content='not json', headers={'Content-Type': 'text/plain'})
            self.check_error(r, 401)
            self.assertEqual(r.headers['www-authenticate'], 'Bearer')
        self.check_error(self.client.get('/batch/jobs/unknown'), 401)
        self.check_error(self.client.delete('/batch/jobs/unknown'), 401)
        for headers in [HEADERS, {'Authorization': 'Bearer ' + KEY}]:
            r = self.client.post('/predict', json=TICKET, headers=headers)
            self.assertEqual(r.status_code, 200)
            validate('predict_response', r.json())
        old = self.app.state.auth
        try:
            self.app.state.auth = APIKeyAuth('')
            self.check_error(self.client.post('/predict', json=TICKET, headers=HEADERS), 401)
        finally:
            self.app.state.auth = old

    def test_validation_and_error_contract(self):
        self.check_error(self.client.post('/predict', content='bad', headers=HEADERS), 415)
        for raw in ['{', 'NaN', '[' * 2000]:
            self.check_error(self.client.post('/predict', content=raw, headers={**HEADERS, 'Content-Type': 'application/json'}), 400)
        for change in [dict(text='  \n'), dict(text=None), dict(text=3), dict(channel='invalid'),
                       dict(subject=None), dict(ticket_id=4), dict(text='a' * 10001), dict(subject='a' * 501)]:
            self.check_error(self.client.post('/predict', json={**TICKET, **change}, headers=HEADERS), 422)
        for payload in [{}, [], None, {'text': 'hello'}, {'channel': 'chat'}]:
            self.check_error(self.client.post('/predict', content=json.dumps(payload), headers={**HEADERS, 'Content-Type': 'application/json'}), 422)
        for path in ['/predict/batch', '/batch/jobs']:
            for tickets in [[], [TICKET, TICKET], [{'channel': 'chat', 'text': 'hello'}], [TICKET, {**TICKET, 'text': ''}]]:
                self.check_error(self.client.post(path, json={'tickets': tickets}, headers=HEADERS), 422)
        for path, size in [('/predict', 1), ('/predict/batch', 5), ('/batch/jobs', 25)]:
            self.check_error(self.client.post(path, content=b' ' * (size * 1024 * 1024 + 1),
                                            headers={**HEADERS, 'Content-Type': 'application/json'}), 413)
        self.check_error(self.client.get('/not-a-route'), 404)
        self.check_error(self.client.get('/predict', headers=HEADERS), 405)
        self.check_error(self.client.get('/batch/jobs/unknown', headers=HEADERS), 404)
        self.check_error(self.client.get('/batch/jobs/unknown/results?limit=0', headers=HEADERS), 422)

    def test_health_unicode_order_and_consistency(self):
        health = self.client.get('/health')
        validate('health_response', health.json())
        examples = ['මගේ බෑගය කාර් එකේ', 'என் ஆர்டர் வரவில்லை', 'mage order eka late', 'en order varala', '😃']
        tickets = [{**TICKET, 'ticket_id': str(i), 'text': text} for i, text in enumerate(examples)]
        response = self.client.post('/predict/batch', json={'tickets': tickets}, headers={**HEADERS, 'X-Request-ID': 'echo-123'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['x-request-id'], 'echo-123')
        validate('batch_response', response.json())
        for ticket, prediction in zip(tickets, response.json()['predictions']):
            single = self.client.post('/predict', json=ticket, headers=HEADERS).json()
            self.assertEqual(single, prediction)
            self.assertEqual(prediction['model_version'], health.json()['model_version'])
        self.check_error(self.client.post('/predict/batch', json={'tickets': [{**TICKET, 'ticket_id': str(i)} for i in range(101)]}, headers=HEADERS), 422)

    def test_bulk_5000_idempotency_pagination_delete(self):
        tickets = [{**TICKET, 'ticket_id': str(i)} for i in range(5000)]
        started = time.perf_counter()
        response = self.client.post('/batch/jobs', json={'tickets': tickets}, headers={**HEADERS, 'Idempotency-Key': 'bulk-test'})
        self.assertEqual(response.status_code, 202, response.text)
        validate('batch_job_status', response.json())
        location = response.headers['location']
        repeat = self.client.post('/batch/jobs', json={'tickets': tickets}, headers={**HEADERS, 'Idempotency-Key': 'bulk-test'})
        self.assertEqual(response.json()['job_id'], repeat.json()['job_id'])
        processed = 0
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            result = self.client.get(location, headers=HEADERS)
            validate('batch_job_status', result.json())
            self.assertGreaterEqual(result.json()['processed'], processed)
            processed = result.json()['processed']
            self.assertEqual(self.client.get('/health').status_code, 200)
            if result.json()['status'] == 'succeeded':
                break
            self.assertNotEqual(result.json()['status'], 'failed')
            time.sleep(0.05)
        self.assertEqual(result.json()['status'], 'succeeded')
        predictions = []
        for offset in range(0, 5000, 777):
            page = self.client.get(location + f'/results?offset={offset}&limit=777', headers=HEADERS)
            validate('batch_job_results', page.json())
            predictions.extend(page.json()['predictions'])
        self.assertEqual([r['ticket_id'] for r in predictions], [r['ticket_id'] for r in tickets])
        self.assertEqual(predictions[0], self.client.post('/predict', json=tickets[0], headers=HEADERS).json())
        print(f'5000-ticket integration run: {time.perf_counter() - started:.2f}s', flush=True)
        self.assertEqual(self.client.delete(location, headers=HEADERS).status_code, 204)
        self.check_error(self.client.get(location, headers=HEADERS), 410)

class BlockingModel:
    version = 'test-model'
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
    def predict(self, tickets):
        self.entered.set()
        self.release.wait(10)
        return [{'ticket_id': t['ticket_id']} for t in tickets]

class JobLifecycleTests(unittest.TestCase):
    def test_capacity_cancel_restart_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'jobs.db'
            model = BlockingModel()
            jobs = Jobs(path, model)
            try:
                first = jobs.submit([TICKET])['job_id']
                self.assertTrue(model.entered.wait(5))
                queued = [jobs.submit([TICKET])['job_id'] for _ in range(3)]
                with self.assertRaises(JobError) as caught:
                    jobs.submit([TICKET])
                self.assertEqual(caught.exception.status, 429)
                with self.assertRaises(JobError) as caught:
                    jobs.results(first, 0, 1)
                self.assertEqual(caught.exception.status, 409)
                jobs.delete(first)
                self.assertEqual(jobs.get(first)['status'], 'cancelled')
                jobs.stop.set()
            finally:
                model.release.set()
                jobs.close()
            jobs = Jobs(path, model)
            try:
                for job_id in queued:
                    state = jobs.get(job_id)
                    self.assertEqual(state['status'], 'failed')
                    self.assertEqual(state['error']['code'], 'interrupted')
                with jobs.connect() as db:
                    row = db.execute('SELECT * FROM jobs WHERE id=?', (queued[0],)).fetchone()
                    self.assertGreaterEqual(row['expires'] - row['finished'], 21600)
                    db.execute('UPDATE jobs SET expires=? WHERE id=?', (time.time() - 1, queued[0]))
                with self.assertRaises(JobError) as caught:
                    jobs.get(queued[0])
                self.assertEqual(caught.exception.status, 410)
            finally:
                jobs.close()

if __name__ == '__main__':
    unittest.main(verbosity=2)
