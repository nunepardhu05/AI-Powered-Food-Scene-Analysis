import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

@contextmanager
def connection():
    path = Path(os.environ.get('NUTRI_DB', 'data/nutriscope.sqlite'))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, body TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS meals (id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE, created_at TEXT NOT NULL, body TEXT NOT NULL)')
    try:
        yield db
        db.commit()
    finally:
        db.close()

def save_profile(profile):
    with connection() as db:
        db.execute('INSERT INTO profiles VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET body=excluded.body', (profile.id, profile.model_dump_json()))

def get_profile(profile_id):
    from .schemas import Profile
    with connection() as db:
        row = db.execute('SELECT body FROM profiles WHERE id=?', (profile_id,)).fetchone()
        return Profile.model_validate_json(row['body']) if row else None

def save_meal(profile_id, body):
    with connection() as db:
        cur = db.execute('INSERT INTO meals(profile_id,created_at,body) VALUES (?,?,?)', (profile_id, datetime.now(timezone.utc).isoformat(), json.dumps(body)))
        return cur.lastrowid

def history(profile_id):
    with connection() as db:
        return [dict(id=r['id'], created_at=r['created_at'], **json.loads(r['body'])) for r in db.execute('SELECT * FROM meals WHERE profile_id=? ORDER BY id DESC', (profile_id,))]

def delete_profile(profile_id):
    with connection() as db:
        db.execute('DELETE FROM profiles WHERE id=?', (profile_id,))
