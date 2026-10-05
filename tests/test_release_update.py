"""Exercise the real release switchover and automatic restoration without sudo."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="The installer uses POSIX shell tools.",
)
REPOSITORY = Path(__file__).resolve().parents[1]
INSTALLER = REPOSITORY / "docker/mantle/install-public.sh"
ROLLBACK = REPOSITORY / "docker/mantle/rollback-public.sh"
OLD_IMAGE = "sha256:" + "a" * 64
NEW_IMAGE = "sha256:" + "b" * 64


@pytest.fixture
def switch_release(tmp_path: Path):
    unit = tmp_path / "live.service"
    release = tmp_path / "release"
    release.mkdir()
    (release / "previous-nzgd_map.service").write_text(OLD_IMAGE)
    (release / "nzgd_map.service").write_text(NEW_IMAGE)
    stub = (
        f"#!{sys.executable}\n"
        + """import json
import os
import shutil
import sys
from pathlib import Path

path = Path(os.environ['NZGD_UPDATE_TEST_STATE'])
state = json.loads(path.read_text())
command = Path(sys.argv[0]).name
args = sys.argv[1:]
unit_image = Path(state['unit']).read_text()
state['calls'].append({'command': command, 'args': args, 'unit': unit_image})
result = 0
if command == 'systemctl':
    action = args[0]
    if action == 'stop':
        state['running'] = False
    elif action == 'start':
        if unit_image == state['new_image'] and state.get('fail_new'):
            result = 23
        elif unit_image == state['old_image'] and state.get('fail_restore'):
            result = 42
        else:
            state['running'] = True
            state['image'] = unit_image
    elif action == 'is-active':
        result = 0 if state['running'] else 3
    elif action not in ['daemon-reload', 'enable', 'reset-failed', 'show']:
        raise AssertionError(args)
elif command == 'docker':
    if args[:2] == ['image', 'inspect']:
        print(state['old_image'])
    elif '{{.Image}} {{.State.Status}} {{.State.Health.Status}}' in args:
        image = state['image']
        if state.get('wrong_new_image') and image == state['new_image']:
            image = 'sha256:unexpected'
        print(image + ' running healthy')
    else:
        print('Container status')
elif command == 'install':
    # Keep the real file replacement, but ownership remains with the test user.
    shutil.copyfile(args[-2], args[-1])
    os.chmod(args[-1], 0o644)
elif command != 'journalctl':
    raise AssertionError(command)
path.write_text(json.dumps(state))
sys.exit(result)
"""
    )
    for command in ["systemctl", "docker", "journalctl", "install"]:
        executable = tmp_path / command
        executable.write_text(stub)
        executable.chmod(0o755)

    # Run the repository's actual transition and EXIT trap. Only privileged
    # preparation is excluded; all file replacements target this test directory.
    source = INSTALLER.read_text()
    exit_function = source.split("deployment_exit() {", 1)[1].split(
        "\ntrap deployment_exit EXIT", 1
    )[0]
    transition = source.split(
        'echo "Starting the public service on Mantle port 7777..."', 1
    )[1]
    manual_rollback = (
        ROLLBACK.read_text()
        .split("rollback_unit=/etc/systemd/system/nzgd_map.service\n", 1)[1]
        .replace(
            "rollback_docker=(sudo -u nzgd_map /usr/bin/docker --host unix:///run/user/1010/docker.sock)",
            "rollback_docker=(docker)",
        )
    )
    state_path = tmp_path / "state.json"

    def invoke(
        *,
        preflight_failure: bool = False,
        initial_install: bool = False,
        manual: bool = False,
        unexpected_unit: bool = False,
        **options,
    ):
        initial_unit = NEW_IMAGE if manual else OLD_IMAGE
        if unexpected_unit:
            initial_unit = "unrelated.service"
        unit.write_text(initial_unit)
        state_path.write_text(
            json.dumps(
                {
                    "calls": [],
                    "unit": str(unit),
                    "old_image": OLD_IMAGE,
                    "new_image": NEW_IMAGE,
                    "image": initial_unit,
                    "running": not initial_install,
                    **options,
                }
            )
        )
        script = "\n".join(
            [
                "set -euo pipefail",
                "deployment_changed_unit=0",
                f"deployment_update={int(not initial_install)}",
                f"deployment_restore_running={int(not initial_install)}",
                "deployment_temporary_database=''",
                f"deployment_release={shlex.quote(str(release))}",
                f"deployment_unit={shlex.quote(str(unit))}",
                "deployment_docker=(docker)",
                f"NZGD_MAP_IMAGE_ID={NEW_IMAGE}",
                f"NZGD_MAP_PREVIOUS_IMAGE_ID={OLD_IMAGE}",
                "deployment_exit() {" + exit_function,
                "trap deployment_exit EXIT",
                "exit 17" if preflight_failure else transition,
            ]
        )
        if manual:
            script = "\n".join(
                [
                    "set -euo pipefail",
                    f"rollback_release={shlex.quote(str(release))}",
                    f"rollback_unit={shlex.quote(str(unit))}",
                    f"NZGD_MAP_PREVIOUS_IMAGE_ID={OLD_IMAGE if not initial_install else ''}",
                    manual_rollback,
                ]
            )
        result = subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                "NZGD_UPDATE_TEST_STATE": str(state_path),
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result, json.loads(state_path.read_text()), unit.read_text()

    return invoke


def test_successful_update_stops_old_service_before_installing_new_unit(
    switch_release: Callable,
):
    result, state, unit = switch_release()
    assert result.returncode == 0, result.stderr
    assert unit == state["image"] == NEW_IMAGE
    assert state["running"]
    stop = next(call for call in state["calls"] if call["args"][0] == "stop")
    assert stop["unit"] == OLD_IMAGE
    start = next(call for call in state["calls"] if call["args"][0] == "start")
    assert start["unit"] == NEW_IMAGE
    assert not any(call["command"] == "journalctl" for call in state["calls"])


def test_preflight_failure_leaves_running_release_untouched(switch_release: Callable):
    result, state, unit = switch_release(preflight_failure=True)
    assert result.returncode == 17
    assert unit == state["image"] == OLD_IMAGE
    assert state["running"]
    assert not state["calls"]


@pytest.mark.parametrize("failure", [{"fail_new": True}, {"wrong_new_image": True}])
def test_failed_update_restores_healthy_previous_image(
    switch_release: Callable, failure: dict[str, bool]
):
    result, state, unit = switch_release(**failure)
    assert result.returncode != 0
    assert unit == state["image"] == OLD_IMAGE
    assert state["running"]
    assert "previous working image is running and healthy again" in result.stderr


def test_failed_restoration_is_reported_without_claiming_success(
    switch_release: Callable,
):
    result, state, unit = switch_release(fail_new=True, fail_restore=True)
    assert result.returncode == 23
    assert unit == OLD_IMAGE
    assert not state["running"]
    assert "Automatic restoration failed" in result.stderr
    assert "healthy again" not in result.stderr


def test_initial_install_failure_still_restores_the_previous_stopped_state(
    switch_release: Callable,
):
    result, state, unit = switch_release(initial_install=True, fail_new=True)
    assert result.returncode == 23
    assert unit == OLD_IMAGE
    assert not state["running"]
    assert [call["unit"] for call in state["calls"] if call["args"][0] == "start"] == [
        NEW_IMAGE
    ]


def test_manual_update_rollback_restarts_and_verifies_the_previous_image(
    switch_release: Callable,
):
    result, state, unit = switch_release(manual=True)
    assert result.returncode == 0, result.stderr
    assert unit == state["image"] == OLD_IMAGE
    assert state["running"]
    assert "previous working image is running and healthy again" in result.stdout


def test_manual_update_rollback_refuses_an_unrelated_unit(switch_release: Callable):
    result, state, unit = switch_release(manual=True, unexpected_unit=True)
    assert result.returncode != 0
    assert unit == "unrelated.service"
    assert state["running"]
    assert not state["calls"]


def test_original_release_manual_rollback_keeps_its_offline_behaviour(
    switch_release: Callable,
):
    result, state, unit = switch_release(manual=True, initial_install=True)
    assert result.returncode == 0, result.stderr
    assert unit == OLD_IMAGE
    assert not state["running"]
    assert "application is stopped" in result.stdout
    assert "healthy again" not in result.stdout
