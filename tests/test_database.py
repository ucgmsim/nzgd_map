"""Database configuration and read-only access checks."""

import sqlite3
from pathlib import Path

import pytest

from nzgd_map import create_app
from nzgd_map.database import open_database


def test_database_is_read_only_and_connection_is_closed(tmp_path: Path):
    path = tmp_path / "database with spaces.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE example (id INTEGER)")
        connection.execute("INSERT INTO example VALUES (1)")
    with open_database(path) as connection:
        assert connection.execute("SELECT id FROM example").fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM example")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_missing_database_is_not_created(tmp_path: Path):
    path = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError), open_database(path):
        pass
    assert not path.exists()


def test_app_uses_configured_secret_and_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("SECRET_KEY", "environment-test-key")
    monkeypatch.setenv("NZGD_DATABASE_PATH", str(tmp_path / "data.db"))
    app = create_app(instance_path=str(tmp_path))
    assert app.secret_key == "environment-test-key"
    assert app.config["DATABASE"] == str(tmp_path / "data.db")


def test_test_config_can_supply_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    app = create_app({"SECRET_KEY": "test-config-key"}, instance_path=str(tmp_path))
    assert app.secret_key == "test-config-key"


def test_secret_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app(instance_path=str(tmp_path))
