"""Open the configured NZGD database without modifying it."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from flask import current_app


@contextmanager
def open_database(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open an existing database read-only and close it on leaving the context."""
    database_path = Path(path if path is not None else current_app.config["DATABASE"])
    connection = sqlite3.connect(
        database_path.expanduser().resolve().as_uri() + "?mode=ro", uri=True
    )
    try:
        yield connection
    finally:
        connection.close()
