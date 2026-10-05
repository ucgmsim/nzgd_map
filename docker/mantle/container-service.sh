#!/usr/bin/env bash
# Invoked by nzgd_map.service as the existing nzgd_map account.
set -euo pipefail
service_release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$service_release/deployment.env"
service_base=/mnt/mantle_data/nzgd_map
service_name=nzgd_map.service
service_label=org.quakecore.nzgd-map.service
service_docker=(docker --host unix:///run/user/1010/docker.sock)

wait_for_docker() {
    local deadline=$((SECONDS + 90))
    until timeout 5 "${service_docker[@]}" info --format '{{.ServerVersion}}' >/dev/null 2>&1; do
        (( SECONDS < deadline )) || { echo "Rootless Docker is not ready." >&2; return 1; }
        sleep 2
    done
}

owned_container() {
    local label
    if ! label=$("${service_docker[@]}" container inspect --format "{{index .Config.Labels \"$service_label\"}}" "$service_name" 2>/dev/null); then
        return 1
    fi
    [[ "$label" == "$service_name" ]] || {
        echo "Container name $service_name is occupied by a container without this service's label." >&2
        return 2
    }
}

cleanup_container() {
    local status=0
    owned_container || status=$?
    case "$status" in
        0) "${service_docker[@]}" container rm --force "$service_name" ;;
        1) return 0 ;;
        *) return "$status" ;;
    esac
}

case "${1:-}" in
    prepare)
        wait_for_docker
        [[ $("${service_docker[@]}" image inspect --format '{{.Id}}' "$NZGD_MAP_IMAGE_ID") == "$NZGD_MAP_IMAGE_ID" ]]
        cleanup_container
        ;;
    run)
        exec "${service_docker[@]}" run --rm --pull never \
            --name "$service_name" --label "$service_label=$service_name" \
            --publish 7777:80 \
            --log-driver local --log-opt max-size=10m --log-opt max-file=3 \
            --env-file "$service_base/runtime/container.env" \
            --env NZGD_DATABASE_PATH=/data/nzgd.db \
            --mount "type=bind,source=$service_base/$NZGD_MAP_DATABASE_NAME,target=/data/nzgd.db,readonly" \
            --mount "type=bind,source=$service_base/$NZGD_MAP_STATIONS_NAME,target=/data/geonet.ll,readonly" \
            --mount "type=bind,source=$service_release/config.py,target=/usr/local/var/nzgd_map-instance/config.py,readonly" \
            "$NZGD_MAP_IMAGE_ID"
        ;;
    healthy)
        deadline=$((SECONDS + 120))
        while (( SECONDS < deadline )); do
            state=$("${service_docker[@]}" container inspect --format '{{.State.Status}} {{.State.Health.Status}}' "$service_name" 2>/dev/null || true)
            case "$state" in
                'running healthy') echo "NZGD application and database checks passed."; exit 0 ;;
                'running unhealthy'|exited*|dead*) echo "NZGD container failed: $state" >&2; exit 1 ;;
            esac
            sleep 2
        done
        echo "NZGD container did not become healthy within 120 seconds." >&2
        exit 1
        ;;
    stop)
        status=0
        owned_container || status=$?
        case "$status" in
            0) "${service_docker[@]}" stop --time 30 "$service_name" ;;
            1) exit 0 ;;
            *) exit "$status" ;;
        esac
        ;;
    cleanup)
        cleanup_container
        ;;
    *) echo "Usage: $0 {prepare|run|healthy|stop|cleanup}" >&2; exit 2 ;;
esac
