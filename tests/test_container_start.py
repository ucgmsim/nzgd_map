"""Verify that the container supervisor stops both servers on failure or shutdown."""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="The deployment supervisor requires a POSIX system and bash.",
)
START_SCRIPT = Path(__file__).resolve().parents[1] / "docker/start.sh"


@pytest.fixture
def server_environment(tmp_path: Path) -> dict[str, str]:
    stub = """#!/usr/bin/env python3
import os
import signal
import sys
import time
from pathlib import Path
name = Path(sys.argv[0]).name
directory = Path(os.environ['NZGD_SUPERVISOR_TEST_DIR'])
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
(directory / (name + '.pid')).write_text(str(os.getpid()))
if name == os.environ.get('NZGD_SUPERVISOR_TEST_FAIL'):
    other = directory / (('nginx' if name == 'uwsgi' else 'uwsgi') + '.pid')
    deadline = time.monotonic() + 5
    while not other.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    sys.exit(int(os.environ.get('NZGD_SUPERVISOR_TEST_CODE', '7')))
signal.pause()
"""
    for name in ["nginx", "uwsgi"]:
        executable = tmp_path / name
        executable.write_text(stub)
        executable.chmod(0o755)
    database = tmp_path / "readable.db"
    database.touch()
    return {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "SECRET_KEY": "local-supervisor-test-only",
        "NZGD_DATABASE_PATH": str(database),
        "NZGD_SUPERVISOR_TEST_DIR": str(tmp_path),
    }


def _wait_for_servers(tmp_path: Path):
    deadline = time.monotonic() + 5
    while not all((tmp_path / f"{name}.pid").exists() for name in ["nginx", "uwsgi"]):
        if time.monotonic() >= deadline:
            pytest.fail("The fake servers did not start")
        time.sleep(0.01)


def _assert_servers_stopped(tmp_path: Path):
    for name in ["nginx", "uwsgi"]:
        pid = int((tmp_path / f"{name}.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("server, code", [("uwsgi", 7), ("nginx", 9), ("nginx", 0)])
def test_server_exit_stops_container_and_other_server(
    server_environment: dict[str, str], tmp_path: Path, server: str, code: int
):
    server_environment.update(
        NZGD_SUPERVISOR_TEST_FAIL=server, NZGD_SUPERVISOR_TEST_CODE=str(code)
    )
    process = subprocess.Popen(
        ["bash", str(START_SCRIPT)],
        env=server_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == (code or 1), (stdout, stderr)
        _assert_servers_stopped(tmp_path)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_container_termination_stops_both_servers(
    server_environment: dict[str, str], tmp_path: Path
):
    process = subprocess.Popen(
        ["bash", str(START_SCRIPT)], env=server_environment, start_new_session=True
    )
    try:
        _wait_for_servers(tmp_path)
        process.terminate()
        assert process.wait(timeout=10) == 143
        _assert_servers_stopped(tmp_path)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
