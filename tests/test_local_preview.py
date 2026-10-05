"""Check local preview isolation and failure cleanup without a Docker daemon."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="The preview uses POSIX shell tools.",
)
PREVIEW = Path(__file__).resolve().parents[1] / "docker/local-preview.sh"
IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def preview_stub(tmp_path: Path):
    executable = tmp_path / "docker"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """import json
import os
import sys
from pathlib import Path

path = Path(os.environ['NZGD_PREVIEW_TEST_STATE'])
state = json.loads(path.read_text())
args = sys.argv[1:]
assert args[:2] == ['--host', 'unix:///var/run/docker.sock']
args = args[2:]
state['calls'].append(args)
result = 0
if args[0] == 'version':
    print('28.1.1')
elif args[0] == 'build':
    pass
elif args[:2] == ['image', 'inspect']:
    print(state['image'])
elif args[:2] == ['network', 'create']:
    state['network'] = 'this-preview-network'
    print(state['network'])
elif args[0] == 'create':
    name = args[args.index('--name') + 1]
    kind = 'proxy' if name.endswith('-proxy') else 'app'
    container_id = 'this-preview-' + kind
    state['containers'][container_id] = {'kind': kind, 'args': args}
    print(container_id)
elif args[0] == 'start':
    container = state['containers'][args[1]]
    if container['kind'] == 'proxy':
        options = container['args']
        binding = options[options.index('--publish') + 1]
        # An older preview already owns 8058. Docker assigns a free port when
        # the caller leaves the host port empty while retaining loopback.
        if binding == '127.0.0.1:8058:80':
            result = 125
        else:
            assert binding == '127.0.0.1::80', binding
    if state.get('fail_start') == container['kind']:
        result = 125
elif args[0] == 'inspect':
    print('healthy')
elif args[0] == 'port':
    assert args[1:] == ['this-preview-proxy', '80/tcp']
    print(state['address'])
elif args[0] == 'logs':
    pass
elif args[0] == 'rm':
    del state['containers'][args[-1]]
elif args[:2] == ['network', 'rm']:
    assert set(state['containers']) == {'older-preview'}
    state['network'] = None
else:
    raise AssertionError(args)
path.write_text(json.dumps(state))
sys.exit(result)
"""
    )
    executable.chmod(0o755)
    # Only replace the HTTP check; keep real Python for session-key generation.
    python = tmp_path / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        + """import json
import os
import sys
from pathlib import Path
if sys.argv[1].endswith('/docker/check_preview.py'):
    path = Path(os.environ['NZGD_PREVIEW_TEST_STATE'])
    state = json.loads(path.read_text())
    state['checked_origins'].append(sys.argv[2])
    path.write_text(json.dumps(state))
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
"""
    )
    python.chmod(0o755)
    database = tmp_path / "database.db"
    database.touch()
    state_path = tmp_path / "state.json"

    def invoke(*options: str, **overrides):
        state = {
            "calls": [],
            "image": IMAGE,
            "address": "127.0.0.1:49157",
            "containers": {"older-preview": {"port": 8058}},
            "network": None,
            "checked_origins": [],
            **overrides,
        }
        state_path.write_text(json.dumps(state))
        result = subprocess.run(
            ["bash", str(PREVIEW), str(database), *options],
            env={
                **os.environ,
                "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                "NZGD_PREVIEW_TEST_STATE": str(state_path),
            },
            input="\n",
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result, json.loads(state_path.read_text())

    return invoke


def test_existing_image_uses_free_loopback_port_and_preserves_old_preview(
    preview_stub: Callable,
):
    result, state = preview_stub("--image", IMAGE, "--keep")
    assert result.returncode == 0, result.stderr
    assert "Private preview: http://127.0.0.1:49157/nzgd" in result.stdout
    assert state["checked_origins"] == ["http://127.0.0.1:49157"]
    assert not any(call[0] == "build" for call in state["calls"])
    creates = [call for call in state["calls"] if call[0] == "create"]
    assert len(creates) == 2
    assert all(IMAGE in call for call in creates)
    assert all(call[call.index("--pull") + 1] == "never" for call in creates)
    assert state["containers"] == {"older-preview": {"port": 8058}}
    assert state["network"] is None


def test_default_preview_builds_then_runs_the_resulting_image(preview_stub: Callable):
    result, state = preview_stub()
    assert result.returncode == 0, result.stderr
    assert len([call for call in state["calls"] if call[0] == "build"]) == 1
    creates = [call for call in state["calls"] if call[0] == "create"]
    assert len(creates) == 2
    assert all(IMAGE in call for call in creates)


@pytest.mark.parametrize("container", ["app", "proxy"])
def test_failed_start_removes_only_this_previews_resources(
    preview_stub: Callable, container: str
):
    result, state = preview_stub("--image", IMAGE, fail_start=container)
    assert result.returncode == 125, result.stderr
    assert state["containers"] == {"older-preview": {"port": 8058}}
    assert state["network"] is None
    assert not state["checked_origins"]


def test_non_loopback_address_is_not_used_for_checks(preview_stub: Callable):
    result, state = preview_stub("--image", IMAGE, address="0.0.0.0:49157")
    assert result.returncode != 0
    assert "Unexpected preview address" in result.stderr
    assert not state["checked_origins"]
    assert state["containers"] == {"older-preview": {"port": 8058}}
    assert state["network"] is None


@pytest.mark.parametrize("options", [("--image",), ("--image", "latest")])
def test_bad_image_option_is_rejected_before_docker_access(
    preview_stub: Callable, options: tuple[str, ...]
):
    result, state = preview_stub(*options)
    assert result.returncode == 2
    assert not state["calls"]
