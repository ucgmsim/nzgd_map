#!/usr/bin/env bash
# Run by the owner with sudo on Mantle. This starts the public NZGD service.
set -euo pipefail
umask 022
[[ $(hostname -s) == mantle && $EUID == 0 ]] || {
    echo "Run this reviewed deployment script with sudo on Mantle." >&2
    exit 1
}
deployment_update=0
case "${1:-}" in
    '') (( $# == 0 )) || exit 2 ;;
    --update) (( $# == 1 )) || exit 2; deployment_update=1 ;;
    *) echo "Usage: $0 [--update]" >&2; exit 2 ;;
esac
deployment_stage=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$deployment_stage"
sha256sum --check SHA256SUMS
source ./deployment.env
deployment_base=/mnt/mantle_data/nzgd_map
deployment_release="$deployment_base/releases/$NZGD_MAP_RELEASE_ID"
deployment_unit=/etc/systemd/system/nzgd_map.service
deployment_database="$deployment_base/$NZGD_MAP_DATABASE_NAME"
deployment_docker=(sudo -u nzgd_map /usr/bin/docker --host unix:///run/user/1010/docker.sock)
deployment_changed_unit=0
deployment_restore_running=0
deployment_temporary_database=""

deployment_exit() {
    local status=$?
    trap - EXIT
    [[ -z "$deployment_temporary_database" ]] || rm -f -- "$deployment_temporary_database"
    if (( status != 0 && deployment_changed_unit )); then
        echo "Startup failed; stopping the app and restoring the previous unit." >&2
        set +e
        systemctl stop nzgd_map.service
        install -m 644 "$deployment_release/previous-nzgd_map.service" "$deployment_unit"
        systemctl daemon-reload
        if (( deployment_restore_running )); then
            systemctl reset-failed nzgd_map.service
            if systemctl start nzgd_map.service \
                && systemctl is-active --quiet nzgd_map.service \
                && [[ $("${deployment_docker[@]}" container inspect --format '{{.Image}} {{.State.Status}} {{.State.Health.Status}}' nzgd_map.service) == "$NZGD_MAP_PREVIOUS_IMAGE_ID running healthy" ]]; then
                echo "The previous working image is running and healthy again." >&2
            else
                echo "Automatic restoration failed; inspect nzgd_map.service before retrying." >&2
            fi
        fi
        journalctl -u nzgd_map.service -n 60 --no-pager
    fi
    exit "$status"
}
trap deployment_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ $(id -u nzgd_map) == 1010 && $(id -g nzgd_map) == 1011 ]]
mountpoint -q /mnt/mantle_data
[[ -r "$deployment_unit" && -r "$deployment_base/$NZGD_MAP_STATIONS_NAME" ]]
"${deployment_docker[@]}" info --format 'Rootless Docker data: {{.DockerRootDir}}'
if (( deployment_update )); then
    [[ ${NZGD_MAP_PREVIOUS_RELEASE_ID:-} =~ ^[0-9]{8}$ \
        && ${NZGD_MAP_PREVIOUS_IMAGE_ID:-} =~ ^sha256:[a-f0-9]{64}$ \
        && "$NZGD_MAP_RELEASE_ID" != "$NZGD_MAP_PREVIOUS_RELEASE_ID" \
        && "$NZGD_MAP_IMAGE_ID" != "$NZGD_MAP_PREVIOUS_IMAGE_ID" ]] || {
        echo "An update needs a distinct release and image plus the expected previous release and image." >&2
        exit 1
    }
    # Refuse to replace an unexpected service or to overwrite its release files.
    cmp -- "$deployment_unit" "$deployment_base/releases/$NZGD_MAP_PREVIOUS_RELEASE_ID/nzgd_map.service"
    [[ $(<"$deployment_unit") != *"$deployment_release/"* ]]
    [[ -z $(systemctl show nzgd_map.service -p DropInPaths --value) ]]
    systemctl is-active --quiet nzgd_map.service
    [[ $("${deployment_docker[@]}" container inspect --format '{{index .Config.Labels "org.quakecore.nzgd-map.service"}} {{.Image}} {{.State.Status}} {{.State.Health.Status}}' nzgd_map.service) == "nzgd_map.service $NZGD_MAP_PREVIOUS_IMAGE_ID running healthy" ]]
    [[ $("${deployment_docker[@]}" image inspect --format '{{.Id}}' "$NZGD_MAP_PREVIOUS_IMAGE_ID") == "$NZGD_MAP_PREVIOUS_IMAGE_ID" ]]
    [[ -f "$deployment_database" && -f "$deployment_base/runtime/container.env" ]]
    deployment_restore_running=1
    echo "Checking the new release while the existing service keeps running."
else
    if systemctl is-active --quiet nzgd_map.service; then
        echo "The application service is already active; use a reviewed update release with --update." >&2
        exit 1
    fi
    [[ -z $(ss -H -ltn 'sport = :7777') ]] || { echo "Port 7777 is already in use." >&2; exit 1; }
    if "${deployment_docker[@]}" container inspect nzgd_map.service >/dev/null 2>&1; then
        echo "A container named nzgd_map.service already exists; review it before deploying." >&2
        exit 1
    fi
fi
python3 - <<'PY'
import shutil

free = shutil.disk_usage('/home/nzgd_map').free
if free < 3 * 1024**3:
    raise SystemExit('Less than 3 GiB free for Docker; review storage before loading the image.')
print(f'System filesystem free before image load: {free:,} bytes')
PY

install -d -o root -g root -m 755 "$deployment_base/releases" "$deployment_release"
if [[ ! -e "$deployment_release/previous-nzgd_map.service" ]]; then
    cp --preserve=mode,timestamps -- "$deployment_unit" "$deployment_release/previous-nzgd_map.service"
else
    cmp -- "$deployment_release/previous-nzgd_map.service" "$deployment_unit"
fi
for filename in container-service.sh install-public.sh rollback-public.sh; do
    install -o root -g root -m 755 "$filename" "$deployment_release/$filename"
done
for filename in config.py deployment.env nzgd_map.service SHA256SUMS; do
    install -o root -g root -m 644 "$filename" "$deployment_release/$filename"
done

if [[ ! -e "$deployment_database" ]]; then
    echo "Installing the versioned database on /mnt/mantle_data..."
    deployment_temporary_database=$(mktemp "$deployment_base/.database-install.XXXXXXXX")
    cp --reflink=auto -- "$NZGD_MAP_DATABASE_NAME" "$deployment_temporary_database"
    chown nzgd_map:nzgd_map "$deployment_temporary_database"
    chmod 644 "$deployment_temporary_database"
    # Publish the finished file without replacing an existing database.
    ln -- "$deployment_temporary_database" "$deployment_database"
    rm -- "$deployment_temporary_database"
    deployment_temporary_database=""
fi
printf '%s  %s\n' "$NZGD_MAP_DATABASE_SHA256" "$deployment_database" | sha256sum --check

install -d -o root -g nzgd_map -m 750 "$deployment_base/runtime"
python3 - "$deployment_base/runtime/container.env" <<'PY'
import os
import secrets
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists() or path.is_symlink():
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0 or metadata.st_gid != 1011 or stat.S_IMODE(metadata.st_mode) != 0o640:
        raise SystemExit('Existing runtime secret file has unexpected ownership or permissions.')
    values = dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)
    if len(values.get('SECRET_KEY', '')) < 32:
        raise SystemExit('Existing runtime file has no suitable session key.')
    print('Reusing the existing private session key.')
else:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, 'w') as stream:
        os.fchown(stream.fileno(), 0, 1011)
        os.fchmod(stream.fileno(), 0o640)
        stream.write('SECRET_KEY=' + secrets.token_hex(32) + '\n')
    print('Created a private session key; its value is not printed.')
PY

if ! "${deployment_docker[@]}" image inspect "$NZGD_MAP_IMAGE_ID" >/dev/null 2>&1; then
    echo "Loading the tested image into the nzgd_map account's Docker daemon..."
    "${deployment_docker[@]}" image load < image.tar.gz
fi
[[ $("${deployment_docker[@]}" image inspect --format '{{.Id}}' "$NZGD_MAP_IMAGE_ID") == "$NZGD_MAP_IMAGE_ID" ]]

# A short, network-disabled process checks the real mounts as the app's user.
# It does not start a server or publish a private preview.
"${deployment_docker[@]}" run --rm --pull never --network none --user 65534:33 \
    --env-file "$deployment_base/runtime/container.env" \
    --env NZGD_DATABASE_PATH=/data/nzgd.db \
    --mount "type=bind,source=$deployment_database,target=/data/nzgd.db,readonly" \
    --mount "type=bind,source=$deployment_base/$NZGD_MAP_STATIONS_NAME,target=/data/geonet.ll,readonly" \
    --mount "type=bind,source=$deployment_release/config.py,target=/usr/local/var/nzgd_map-instance/config.py,readonly" \
    --entrypoint python "$NZGD_MAP_IMAGE_ID" -c '
from nzgd_map import create_app
from nzgd_map.views import load_geonet_stations
app = create_app()
assert app.config["SESSION_COOKIE_SECURE"] is True
assert app.config["LAST_NZGD_RETRIEVAL_DATE"] == "9 July 2026"
assert app.test_client().get("/query_help").status_code == 200
# The station loader checks the session before falling back to the default file.
with app.test_request_context():
    stations = load_geonet_stations()
    assert not stations.empty
    print(f"Database and production config readable; {len(stations)} GeoNet stations loaded.")
'
systemd-analyze verify "$deployment_release/nzgd_map.service"

echo "Starting the public service on Mantle port 7777..."
deployment_changed_unit=1
if (( deployment_update )); then
    systemctl stop nzgd_map.service
fi
install -o root -g root -m 644 "$deployment_release/nzgd_map.service" "$deployment_unit"
systemctl daemon-reload
systemctl enable nzgd_map.service
systemctl reset-failed nzgd_map.service
systemctl start nzgd_map.service
systemctl is-active --quiet nzgd_map.service
[[ $("${deployment_docker[@]}" container inspect --format '{{.Image}} {{.State.Status}} {{.State.Health.Status}}' nzgd_map.service) == "$NZGD_MAP_IMAGE_ID running healthy" ]]
"${deployment_docker[@]}" container inspect --format 'Container: {{.Name}}; image: {{.Image}}; health: {{.State.Health.Status}}' nzgd_map.service
systemctl show nzgd_map.service -p ActiveState -p SubState -p UnitFileState
deployment_changed_unit=0
echo "Mantle startup passed. Check https://quakecoresoft.canterbury.ac.nz/nzgd through the public route."
