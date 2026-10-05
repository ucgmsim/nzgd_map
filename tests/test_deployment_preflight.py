"""Run the installer's real inline application check without Docker or sudo."""

import shlex
from pathlib import Path

import pytest

from nzgd_map import create_app

INSTALLER = Path(__file__).resolve().parents[1] / "docker/mantle/install-public.sh"


@pytest.mark.parametrize("has_stations", [True, False])
def test_installer_checks_default_stations_in_a_request_context(
    database_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    has_stations: bool,
):
    station_file = tmp_path / "default-stations.ll"
    station_file.write_text("172.63 -43.53 DEPLOYMENT_TEST\n" if has_stations else "")
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "deployment-test-only",
            "DATABASE": str(database_path),
            "GEONET_STATIONS_PATH": str(station_file),
            "LAST_NZGD_RETRIEVAL_DATE": "9 July 2026",
            "SESSION_COOKIE_SECURE": True,
        },
        instance_path=str(tmp_path / "instance"),
    )
    monkeypatch.setattr("nzgd_map.create_app", lambda: app)
    # Execute the actual Python passed to the diagnostic container, so shell
    # edits cannot silently bypass or duplicate the application check in tests.
    command = INSTALLER.read_text().split(
        '--entrypoint python "$NZGD_MAP_IMAGE_ID" -c ', 1
    )[1]
    arguments = shlex.shlex(command, posix=True)
    arguments.whitespace_split = True
    code = compile(next(arguments), str(INSTALLER), "exec")
    if has_stations:
        exec(code, {})  # noqa: S102 - execute the repository-owned diagnostic verbatim.
        assert "1 GeoNet stations loaded" in capsys.readouterr().out
    else:
        with pytest.raises(AssertionError):
            exec(code, {})  # noqa: S102 - execute the repository-owned diagnostic verbatim.
