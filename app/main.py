import json
import os
import anyio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.concurrency import run_in_threadpool
from jsonschema import Draft202012Validator
import yaml
from app.model import Model
from app.jobs import Jobs, JobError
from app.auth import APIKeyAuth

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = {name: Draft202012Validator(json.loads((ROOT / 'specs' / (name + '.schema.json')).read_text(encoding='utf-8')))
           for name in ['predict_request', 'batch_request', 'batch_job_request']}

def error(status, code, message, details=None, headers=None):
    body = dict(code=code, message=message)
    if details:
        body['details'] = details
    return JSONResponse({'error': body}, status_code=status, headers=headers)

class InputError(Exception):
    def __init__(self, status, code, message, details=None):
        self.status, self.code, self.message, self.details = status, code, message, details

async def body(request, schema, megabytes):
    if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'application/json':
        raise InputError(415, 'unsupported_media_type', 'Content-Type must be application/json.')
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > megabytes * 1024 * 1024:
            raise InputError(413, 'payload_too_large', 'Request body exceeds the size limit.')
    try:
        parsed = json.loads(data.decode('utf-8'), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError, RecursionError):
        raise InputError(400, 'invalid_json', 'Request body must be valid UTF-8 JSON.')
    details = []
    for issue in SCHEMAS[schema].iter_errors(parsed):
        path = list(issue.absolute_path)
        detail = dict(field=str(path[-1]) if path else 'body', issue=issue.validator + ' constraint failed')
        if len(path) > 1 and path[0] == 'tickets' and isinstance(path[1], int):
            detail['index'] = path[1]
        details.append(detail)
        if len(details) >= 100:
            break
    if details:
        raise InputError(422, 'validation_error', 'Request does not match the schema.', details)
    if schema != 'predict_request':
        seen = set()
        for i, ticket in enumerate(parsed['tickets']):
            if ticket['ticket_id'] in seen:
                raise InputError(422, 'validation_error', 'Duplicate ticket_id.',
                                 [dict(index=i, field='ticket_id', issue='must be unique')])
            seen.add(ticket['ticket_id'])
    return parsed

def create_app(model_path=None, db_path=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.auth = APIKeyAuth(os.environ.get('API_KEY', ''))
        app.state.inference_limiter = anyio.CapacityLimiter(4)
        app.state.model = Model(model_path or os.environ.get('MODEL_PATH', ROOT / 'models' / 'baseline.joblib'))
        app.state.jobs = Jobs(db_path or os.environ.get('JOB_DB_PATH', ROOT / 'runtime' / 'jobs.sqlite3'), app.state.model)
        try:
            yield
        finally:
            await run_in_threadpool(app.state.jobs.close)

    app = FastAPI(title='Team Hydra — TensorForge API', lifespan=lifespan)
    def openapi():
        document = yaml.safe_load((ROOT / 'specs' / 'tensorforge-phase2-openapi-v2.yaml').read_text(encoding='utf-8'))
        document['info']['title'] = 'Team Hydra — TensorForge Support Ticket API'
        return document
    app.openapi = openapi
    app.mount('/ui', StaticFiles(directory=ROOT / 'web'), name='ui')

    @app.get('/', include_in_schema=False)
    async def dashboard():
        return FileResponse(ROOT / 'web' / 'index.html', headers={
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            'Referrer-Policy': 'no-referrer', 'X-Content-Type-Options': 'nosniff'})

    @app.middleware('http')
    async def authorize(request, call_next):
        path = request.url.path
        if path == '/predict' or path.startswith('/predict/') or path == '/batch/jobs' or path.startswith('/batch/jobs/'):
            auth = getattr(app.state, 'auth', None)
            if auth is None or not auth.accepts(request.headers):
                response = error(401, 'unauthorized', 'Missing or invalid API key.', headers={'WWW-Authenticate': 'Bearer'})
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        if 'x-request-id' in request.headers:
            response.headers['X-Request-ID'] = request.headers['x-request-id']
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(InputError)
    async def input_error(request, exc):
        return error(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(JobError)
    async def job_error(request, exc):
        return error(exc.status, exc.code, exc.message, headers={'Retry-After': '3'} if exc.status == 429 else None)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error(exc.status_code, 'http_error', str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return error(422, 'validation_error', 'Invalid request parameters.')

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        return error(500, 'internal_error', 'An internal error occurred.')

    @app.get('/health')
    async def health():
        model = getattr(app.state, 'model', None)
        return JSONResponse(dict(status='ok' if model else 'loading', model_version=model.version if model else None,
                                 model_loaded=model is not None), status_code=200 if model else 503)

    @app.post('/predict')
    async def predict(request: Request):
        ticket = await body(request, 'predict_request', 1)
        return (await anyio.to_thread.run_sync(app.state.model.predict, [ticket], limiter=app.state.inference_limiter))[0]

    @app.post('/predict/batch')
    async def batch(request: Request):
        payload = await body(request, 'batch_request', 5)
        return {'predictions': await anyio.to_thread.run_sync(app.state.model.predict, payload['tickets'], limiter=app.state.inference_limiter)}

    @app.post('/batch/jobs')
    async def submit(request: Request):
        payload = await body(request, 'batch_job_request', 25)
        status = await run_in_threadpool(app.state.jobs.submit, payload['tickets'], request.headers.get('idempotency-key'))
        return JSONResponse(status, status_code=202, headers={'Location': '/batch/jobs/' + status['job_id'], 'Retry-After': '3'})

    @app.get('/batch/jobs/{job_id}')
    def status(job_id: str):
        result = app.state.jobs.get(job_id)
        return JSONResponse(result, headers={'Retry-After': '3'} if result['status'] in ('queued', 'running') else {})

    @app.get('/batch/jobs/{job_id}/results')
    def results(job_id: str, offset: int = 0, limit: int = 5000):
        if offset < 0 or not 1 <= limit <= 5000:
            raise InputError(422, 'validation_error', 'offset must be nonnegative; limit must be 1 to 5000.')
        return app.state.jobs.results(job_id, offset, limit)

    @app.delete('/batch/jobs/{job_id}')
    def delete(job_id: str):
        app.state.jobs.delete(job_id)
        return Response(status_code=204)

    return app

app = create_app()
