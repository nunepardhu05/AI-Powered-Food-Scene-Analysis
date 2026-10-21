import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('NUTRI_DB',str(tmp_path/'test.sqlite'))
    with TestClient(app) as c: yield c
