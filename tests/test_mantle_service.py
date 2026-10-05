"""Exercise deployment lifecycle decisions without accessing a Docker daemon."""

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="The deployment service uses POSIX shell tools.",
)
RELEASE = Path(__file__).resolve().parents[1] / "docker/mantle"
IMAGE_ID = dict(
    line.split("=", 1)
    for line in (RELEASE / "deployment.env").read_text().splitlines()
    if line and not line.startswith("#")
)["NZGD_MAP_IMAGE_ID"]


@pytest.fixture
def service_stub(tmp_path: Path):
    executable = tmp_path / "docker"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

path = Path(os.environ['NZGD_MANTLE_TEST_STATE'])
state = json.loads(path.read_text())
args = sys.argv[1:]
assert args[:2] == ['--host', 'unix:///run/user/1010/docker.sock']
args = args[2:]
state['calls'].append(args)
result = 0
if args[0] == 'info':
    if state.get('daemon_delays', 0):
        state['daemon_delays'] -= 1
        result = 1
elif args[:2] == ['image', 'inspect']:
    print(state['image'])
elif args[:2] == ['container', 'inspect']:
    if 'Labels' in args[3]:
        if state['owner'] is None:
            result = 1
        else:
            print(state['owner'])
    else:
        print(state['health'].pop(0))
elif args[:2] == ['container', 'rm']:
    state['owner'] = None
elif args[0] in ['run', 'stop']:
    pass
else:
    raise AssertionError(args)
path.write_text(json.dumps(state))
sys.exit(result)
"""
    )
    executable.chmod(0o755)
    sleeper = tmp_path / "sleep"
    sleeper.write_text("#!/bin/sh\nexit 0\n")
    sleeper.chmod(0o755)
    state_path = tmp_path / "state.json"

    def invoke(action: str, **overrides):
        state = {
            "calls": [],
            "image": IMAGE_ID,
            "owner": None,
            "health": ["running healthy"],
            **overrides,
        }
        state_path.write_text(json.dumps(state))
        environment = {
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "NZGD_MANTLE_TEST_STATE": str(state_path),
        }
        result = subprocess.run(
            ["bash", str(RELEASE / "container-service.sh"), action],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result, json.loads(state_path.read_text())

    return invoke


def test_delayed_docker_is_retried(service_stub: Callable):
    result, state = service_stub("prepare", daemon_delays=2)
    assert result.returncode == 0, result.stderr
    assert len([call for call in state["calls"] if call[0] == "info"]) == 3


def test_wrong_image_does_not_remove_any_container(service_stub: Callable):
    result, state = service_stub(
        "prepare", image="sha256:wrong", owner="nzgd_map.service"
    )
    assert result.returncode != 0
    assert state["owner"] == "nzgd_map.service"
    assert not any(call[:2] == ["container", "rm"] for call in state["calls"])


@pytest.mark.parametrize("action", ["prepare", "cleanup", "stop"])
def test_name_collision_is_left_untouched(service_stub: Callable, action: str):
    result, state = service_stub(action, owner="another-service")
    assert result.returncode != 0
    assert state["owner"] == "another-service"
    assert not any(
        call[0] == "stop" or call[:2] == ["container", "rm"] for call in state["calls"]
    )


def test_previous_owned_container_is_cleaned_before_start(service_stub: Callable):
    result, state = service_stub("prepare", owner="nzgd_map.service")
    assert result.returncode == 0, result.stderr
    assert state["owner"] is None


def test_run_uses_pinned_image_read_only_mounts_and_private_key_file(
    service_stub: Callable,
):
    result, state = service_stub("run")
    assert result.returncode == 0, result.stderr
    (command,) = state["calls"]
    assert command[-1] == IMAGE_ID
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--publish") + 1] == "7777:80"
    assert (
        command[command.index("--env-file") + 1]
        == "/mnt/mantle_data/nzgd_map/runtime/container.env"
    )
    mounts = [command[i + 1] for i, part in enumerate(command) if part == "--mount"]
    assert len(mounts) == 3
    assert all(mount.endswith(",readonly") for mount in mounts)
    assert any("target=/data/nzgd.db," in mount for mount in mounts)
    assert any("target=/data/geonet.ll," in mount for mount in mounts)
    assert any(
        "target=/usr/local/var/nzgd_map-instance/config.py," in mount
        for mount in mounts
    )


def test_health_waits_for_a_successful_application_check(service_stub: Callable):
    result, state = service_stub(
        "healthy", health=["", "running starting", "running healthy"]
    )
    assert result.returncode == 0, result.stderr
    assert state["health"] == []


@pytest.mark.parametrize(
    "health", ["running unhealthy", "exited unhealthy", "dead starting"]
)
def test_failed_application_health_fails_startup(service_stub: Callable, health: str):
    result, _ = service_stub("healthy", health=[health])
    assert result.returncode != 0
    assert "failed" in result.stderr


def test_stop_only_stops_the_labeled_application(service_stub: Callable):
    result, state = service_stub("stop", owner="nzgd_map.service")
    assert result.returncode == 0, result.stderr
    assert state["calls"][-1] == ["stop", "--time", "30", "nzgd_map.service"]
