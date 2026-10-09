"""Persistent, single-process job service. One worker, three waiting slots."""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

RETENTION = 6 * 60 * 60

class JobError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message

def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp is not None else None

class Jobs:
    def __init__(self, path, model):
        self.path, self.model = str(path), model
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # A process lock prevents a second server from corrupting restart recovery.
        self.lock = open(self.path + '.lock', 'a+b')
        self.lock.seek(0)
        if self.lock.read(1) == b'':
            self.lock.write(b'0')
            self.lock.flush()
        self.lock.seek(0)
        try:
            if __import__('os').name == 'nt':
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError('Use one API process per job database; multiple workers are unsupported.')
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, idem TEXT UNIQUE, status TEXT NOT NULL,
                total INTEGER, processed INTEGER, created REAL, started REAL,
                finished REAL, expires REAL, version TEXT, error TEXT, tickets TEXT, results TEXT)''')
            now = time.time()
            db.execute("UPDATE jobs SET status='failed', finished=?, expires=?, error=?, tickets=NULL WHERE status IN ('queued','running')",
                       (now, now + RETENTION, json.dumps(dict(code='interrupted', message='Service restarted before completion.'))))
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name='batch-worker')
        self.thread.start()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def expire(self, db):
        db.execute('UPDATE jobs SET tickets=NULL, results=NULL WHERE expires <= ?', (time.time(),))

    def get_row(self, db, job_id):
        row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise JobError(404, 'not_found', 'Unknown job_id.')
        if row['expires'] is not None and row['expires'] <= time.time():
            raise JobError(410, 'expired', 'Job has expired or was discarded.')
        return row

    @staticmethod
    def status(row):
        result = dict(job_id=row['id'], status=row['status'], total=row['total'], processed=row['processed'],
                      model_version=row['version'], created_at=iso(row['created']), started_at=iso(row['started']),
                      finished_at=iso(row['finished']), expires_at=iso(row['expires']))
        if row['error']:
            result['error'] = json.loads(row['error'])
        return result

    def submit(self, tickets, idem=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self.expire(db)
            if idem is not None:
                row = db.execute('SELECT * FROM jobs WHERE idem=?', (idem,)).fetchone()
                if row is not None:
                    return self.status(self.get_row(db, row['id']))
            active = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if active >= 4:
                raise JobError(429, 'queue_full', 'One running and three waiting jobs are already accepted.')
            job_id = str(uuid.uuid4())
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (job_id, idem, 'queued', len(tickets), 0, time.time(), None, None, None,
                        self.model.version, None, json.dumps(tickets, ensure_ascii=False), None))
            result = self.status(self.get_row(db, job_id))
        self.wake.set()
        return result

    def get(self, job_id):
        with self.connect() as db:
            return self.status(self.get_row(db, job_id))

    def results(self, job_id, offset, limit):
        with self.connect() as db:
            row = self.get_row(db, job_id)
            if row['status'] != 'succeeded':
                raise JobError(409, 'not_ready', 'Job has not succeeded.')
            predictions = json.loads(row['results'])
        return dict(job_id=job_id, status='succeeded', total=row['total'], offset=offset, limit=limit,
                    next_offset=offset + limit if offset + limit < row['total'] else None,
                    model_version=row['version'], predictions=predictions[offset:offset + limit])

    def delete(self, job_id):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self.get_row(db, job_id)
            now = time.time()
            if row['status'] in ('queued', 'running'):
                db.execute("UPDATE jobs SET status='cancelled', finished=?, expires=?, tickets=NULL, results=NULL WHERE id=?",
                           (now, now + RETENTION, job_id))
            else:
                db.execute('UPDATE jobs SET expires=?, tickets=NULL, results=NULL WHERE id=?', (now, job_id))

    def run(self):
        while not self.stop.is_set():
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                self.expire(db)
                row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                if row:
                    db.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), row['id']))
            if not row:
                self.wake.wait(0.5)
                self.wake.clear()
                continue
            try:
                tickets = json.loads(row['tickets'])
                predictions = []
                for start in range(0, len(tickets), 100):
                    if self.stop.is_set():
                        return  # Recovery marks unfinished jobs interrupted on next start.
                    with self.connect() as db:
                        state = db.execute('SELECT status FROM jobs WHERE id=?', (row['id'],)).fetchone()[0]
                    if state != 'running':
                        break
                    predictions.extend(self.model.predict(tickets[start:start + 100]))
                    with self.connect() as db:
                        db.execute("UPDATE jobs SET processed=? WHERE id=? AND status='running'", (len(predictions), row['id']))
                else:
                    now = time.time()
                    with self.connect() as db:
                        db.execute("UPDATE jobs SET status='succeeded', finished=?, expires=?, results=?, tickets=NULL WHERE id=? AND status='running'",
                                   (now, now + RETENTION, json.dumps(predictions, ensure_ascii=False), row['id']))
            except Exception:
                now = time.time()
                with self.connect() as db:
                    db.execute("UPDATE jobs SET status='failed', error=?, finished=?, expires=?, tickets=NULL WHERE id=? AND status='running'",
                               (json.dumps(dict(code='inference_failed', message='Batch inference failed.')),
                                now, now + RETENTION, row['id']))

    def close(self):
        self.stop.set()
        self.wake.set()
        self.thread.join()
        if __import__('os').name == 'nt':
            import msvcrt
            self.lock.seek(0)
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
        self.lock.close()
