"""Run the container's uWSGI configuration against requests sized like proxied ones."""

import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

NZGD_INI = Path(__file__).resolve().parents[1] / "docker/nzgd.ini"
UWSGI = next(
    (
        str(path)
        for path in [Path(sys.executable).with_name("uwsgi"), shutil.which("uwsgi")]
        if path and os.access(path, os.X_OK)
    ),
    None,
)
pytestmark = pytest.mark.skipif(UWSGI is None, reason="uWSGI is not installed.")


@pytest.fixture
def uwsgi_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    (tmp_path / "probe_app.py").write_text(
        "def application(environ, start_response):\n"
        "    start_response('200 OK', [('Content-Type', 'text/plain')])\n"
        "    return [environ['PATH_INFO'].encode()]\n"
    )
    # Keep every deployed setting except the socket, user switch and app.
    ini = NZGD_INI.read_text()
    for key, line in {
        "socket": "socket = nzgd.sock",
        "uid": "",
        "gid": "",
        "module": "module = probe_app",
        "callable": "callable = application",
    }.items():
        ini, count = re.subn(rf"(?m)^{key}\s*=.*$", line, ini)
        assert count == 1, f"docker/nzgd.ini no longer sets {key}"
    (tmp_path / "nzgd.ini").write_text(f"{ini}\npythonpath = {tmp_path}\n")
    # Socket paths are limited to 108 bytes, so bind and connect relatively.
    monkeypatch.chdir(tmp_path)
    server = subprocess.Popen([UWSGI, "--ini", "nzgd.ini", "--logto", "uwsgi.log"])
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "nzgd.sock").exists():
            if server.poll() is not None or time.monotonic() > deadline:
                pytest.fail((tmp_path / "uwsgi.log").read_text())
            time.sleep(0.05)
        yield "nzgd.sock"
    finally:
        server.terminate()
        server.wait(timeout=10)


def _uwsgi_request(path: str, variables: dict[str, str]) -> bytes:
    """Send one request in the uwsgi protocol, as nginx's uwsgi_pass does."""
    body = b"".join(
        struct.pack("<H", len(key)) + key + struct.pack("<H", len(value)) + value
        for key, value in ((k.encode(), v.encode()) for k, v in variables.items())
    )
    response = b""
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(10)
        client.connect(path)
        client.sendall(struct.pack("<BHB", 0, len(body), 0) + body)
        try:
            while chunk := client.recv(65536):
                response += chunk
        except ConnectionResetError:
            pass  # uWSGI resets requests it rejects; the assertion reports it.
    return response


@pytest.mark.parametrize("header_bytes", [4_500, 60_000])
def test_requests_larger_than_default_uwsgi_buffer_are_served(
    uwsgi_socket: str, header_bytes: int
):
    # Proxied browser requests can carry about 4 KB of headers before the query
    # string. uWSGI's default 4 KB buffer rejects larger requests, which nginx
    # reports as 502. 60 KB approaches the uwsgi protocol's 64 KB packet limit.
    query = "query=deepest_depth+%3E+3&vs30_availability=all"
    response = _uwsgi_request(
        uwsgi_socket,
        {
            "REQUEST_METHOD": "GET",
            "SCRIPT_NAME": "/nzgd",
            "PATH_INFO": "/nzgd/",
            "REQUEST_URI": f"/nzgd/?{query}",
            "QUERY_STRING": query,
            "SERVER_PROTOCOL": "HTTP/1.0",
            "SERVER_NAME": "localhost",
            "SERVER_PORT": "80",
            "HTTP_HOST": "mantle.canterbury.ac.nz:7777",
            "HTTP_X_PADDING": "a" * header_bytes,
        },
    )
    assert response.startswith(b"HTTP/1.0 200 OK"), response
    # The deployed fixpathinfo route strips the /nzgd mount from PATH_INFO.
    assert response.endswith(b"\r\n\r\n/")
