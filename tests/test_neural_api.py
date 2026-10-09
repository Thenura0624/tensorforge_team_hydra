"""Optional real saved-neural-model integration test (no external downloads)."""
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from app.main import create_app, ROOT

@unittest.skipUnless(os.environ.get('NEURAL_TEST_MODEL'), 'Set NEURAL_TEST_MODEL to a saved tournament model directory.')
class NeuralAPITests(unittest.TestCase):
    def test_saved_model_all_prediction_routes(self):
        def validate(name, payload):
            Draft202012Validator(json.loads((ROOT / 'specs' / (name + '.schema.json')).read_text())).validate(payload)
        assigned = secrets.token_urlsafe(32)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'API_KEY': assigned, 'MODEL_DEVICE': 'cpu'}):
            app = create_app(model_path=os.environ['NEURAL_TEST_MODEL'], db_path=Path(directory) / 'jobs.db')
            with TestClient(app) as client:
                health = client.get('/health')
                self.assertEqual(health.status_code, 200)
                validate('health_response', health.json())
                headers = {'X-API-Key': assigned}
                tickets = [dict(ticket_id=str(i), channel='chat', text=text) for i, text in enumerate(
                    ['My ride was charged twice.', 'මගේ බෑග් එක කාර් එකේ අමතක වුණා', 'என் ஆர்டர் வரவில்லை'])]
                single = client.post('/predict', json=tickets[0], headers=headers)
                self.assertEqual(single.status_code, 200)
                validate('predict_response', single.json())
                self.assertEqual(single.json()['model_version'], health.json()['model_version'])
                batch = client.post('/predict/batch', json={'tickets': tickets}, headers=headers)
                self.assertEqual(batch.status_code, 200)
                validate('batch_response', batch.json())
                self.assertEqual(batch.json()['predictions'][0]['category'], single.json()['category'])
                job = client.post('/batch/jobs', json={'tickets': tickets}, headers=headers)
                self.assertEqual(job.status_code, 202)
                location = job.headers['location']
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    status = client.get(location, headers=headers)
                    validate('batch_job_status', status.json())
                    if status.json()['status'] in ('succeeded', 'failed'):
                        break
                    self.assertEqual(client.get('/health').status_code, 200)
                    time.sleep(0.05)
                self.assertEqual(status.json()['status'], 'succeeded')
                result = client.get(location + '/results', headers=headers)
                validate('batch_job_results', result.json())
                self.assertEqual(result.json()['predictions'], batch.json()['predictions'])
                self.assertEqual(client.delete(location, headers=headers).status_code, 204)
                self.assertEqual(client.get(location, headers=headers).status_code, 410)

if __name__ == '__main__':
    unittest.main(verbosity=2)
