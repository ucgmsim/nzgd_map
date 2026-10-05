#!/usr/bin/env bash
# Restore the previous unit; update releases also restart the previous image.
set -euo pipefail
[[ $(hostname -s) == mantle && $EUID == 0 ]] || {
    echo "Run this rollback with sudo on Mantle." >&2
    exit 1
}
rollback_release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$rollback_release/deployment.env"
[[ -f "$rollback_release/previous-nzgd_map.service" ]] || {
    echo "No saved previous unit is available here." >&2
    exit 1
}
rollback_unit=/etc/systemd/system/nzgd_map.service
rollback_docker=(sudo -u nzgd_map /usr/bin/docker --host unix:///run/user/1010/docker.sock)
rollback_restart=0
if [[ -n ${NZGD_MAP_PREVIOUS_IMAGE_ID:-} ]]; then
    [[ "$NZGD_MAP_PREVIOUS_IMAGE_ID" =~ ^sha256:[a-f0-9]{64}$ ]]
    # Accept this release's unit or an already restored backup, never a later
    # deployment or an unrelated service configuration.
    cmp -s -- "$rollback_unit" "$rollback_release/nzgd_map.service" \
        || cmp -- "$rollback_unit" "$rollback_release/previous-nzgd_map.service"
    [[ -z $(systemctl show nzgd_map.service -p DropInPaths --value) ]]
    [[ $("${rollback_docker[@]}" image inspect --format '{{.Id}}' "$NZGD_MAP_PREVIOUS_IMAGE_ID") == "$NZGD_MAP_PREVIOUS_IMAGE_ID" ]]
    rollback_restart=1
fi
systemctl stop nzgd_map.service
install -o root -g root -m 644 "$rollback_release/previous-nzgd_map.service" "$rollback_unit"
systemctl daemon-reload
if (( rollback_restart )); then
    systemctl reset-failed nzgd_map.service
    systemctl start nzgd_map.service
    systemctl is-active --quiet nzgd_map.service
    [[ $("${rollback_docker[@]}" container inspect --format '{{.Image}} {{.State.Status}} {{.State.Health.Status}}' nzgd_map.service) == "$NZGD_MAP_PREVIOUS_IMAGE_ID running healthy" ]]
    echo "The previous working image is running and healthy again."
else
    echo "Previous service configuration restored; the application is stopped."
fi
echo "Images, database files, and the session key have been retained."
