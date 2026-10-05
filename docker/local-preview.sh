#!/usr/bin/env bash
# Build and test on the workstation only. Nothing is sent to Mantle or 2p.
set -euo pipefail

case "$(hostname -s)" in
    mantle|ucquakecore2p|2p)
        echo "Run this preview on your workstation, not on Mantle or 2p." >&2
        exit 1
        ;;
esac
usage() {
    echo "Usage: bash docker/local-preview.sh /absolute/path/to/database.db [--keep] [--image sha256:IMAGE_ID]" >&2
    exit 2
}
(( $# >= 1 )) || usage
preview_db=$(realpath -e -- "$1")
[[ -f "$preview_db" && -r "$preview_db" ]] || { echo "Cannot read the database." >&2; exit 1; }
shift
preview_keep=""
preview_image=""
while (( $# )); do
    case "$1" in
        --keep)
            preview_keep=--keep
            shift
            ;;
        --image)
            (( $# >= 2 )) || usage
            [[ "$2" =~ ^sha256:[a-f0-9]{64}$ ]] || usage
            preview_image="$2"
            shift 2
            ;;
        *) usage ;;
    esac
done
preview_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Explicitly use this workstation's Unix socket, regardless of Docker context
# or DOCKER_HOST. Sudo, if needed, prompts only in the owner's own terminal.
preview_docker=(docker --host unix:///var/run/docker.sock)
if ! "${preview_docker[@]}" version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    preview_docker=(sudo docker --host unix:///var/run/docker.sock)
    "${preview_docker[@]}" version --format '{{.Server.Version}}'
fi

if [[ -z "$preview_image" ]]; then
    preview_tag="nzgd-map:preview-20260923"
    "${preview_docker[@]}" build --tag "$preview_tag" --file "$preview_root/docker/Dockerfile" "$preview_root"
    preview_image=$("${preview_docker[@]}" image inspect --format '{{.Id}}' "$preview_tag")
fi
"${preview_docker[@]}" image inspect --format 'Image: {{.Id}}; size: {{.Size}} bytes' "$preview_image"

preview_network=""
preview_backend=""
preview_proxy=""
preview_tmp=$(mktemp -d -t nzgd-map-preview.XXXXXXXX)
cleanup() {
    local preview_status=$?
    trap - EXIT INT TERM
    if (( preview_status != 0 )); then
        [[ -z "$preview_backend" ]] || "${preview_docker[@]}" logs --tail 60 "$preview_backend" || true
        [[ -z "$preview_proxy" ]] || "${preview_docker[@]}" logs --tail 30 "$preview_proxy" || true
    fi
    [[ -z "$preview_proxy" ]] || "${preview_docker[@]}" rm -f "$preview_proxy" >/dev/null || true
    [[ -z "$preview_backend" ]] || "${preview_docker[@]}" rm -f "$preview_backend" >/dev/null || true
    [[ -z "$preview_network" ]] || "${preview_docker[@]}" network rm "$preview_network" >/dev/null || true
    rm -rf -- "$preview_tmp"
    exit "$preview_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Scope cookies to this application; plain HTTP is used only for this private
# local rehearsal. Production uses docker/config.py.example with secure cookies.
printf '%s\n' \
    'SESSION_COOKIE_NAME = "nzgd_session"' \
    'SESSION_COOKIE_PATH = "/nzgd"' \
    'SESSION_COOKIE_HTTPONLY = True' \
    'SESSION_COOKIE_SAMESITE = "Lax"' > "$preview_tmp/config.py"
chmod 644 "$preview_tmp/config.py"
# Sudo normally removes exported variables. Pass a private file to the Docker
# client instead, so the key survives sudo without appearing in its arguments.
(
    umask 077
    python3 -c 'import secrets; print("SECRET_KEY=" + secrets.token_hex(32))' > "$preview_tmp/environment"
)

preview_network=$("${preview_docker[@]}" network create "nzgd-map-preview-$$")
# Save each created container's ID before starting it so a startup failure
# still cleans up exactly this invocation's containers.
preview_backend=$("${preview_docker[@]}" create --pull never \
    --name "nzgd-map-preview-$$-app" \
    --network "$preview_network" --network-alias nzgd-backend \
    --env-file "$preview_tmp/environment" \
    --mount "type=bind,source=$preview_db,target=/data/nzgd.db,readonly" \
    --mount "type=bind,source=$preview_tmp/config.py,target=/usr/local/var/nzgd_map-instance/config.py,readonly" \
    "$preview_image")
"${preview_docker[@]}" start "$preview_backend" >/dev/null
echo "Waiting for the application and database checks..."
preview_health="starting"
for ((attempt=0; attempt<60; attempt++)); do
    preview_health=$("${preview_docker[@]}" inspect --format '{{.State.Health.Status}}' "$preview_backend")
    [[ "$preview_health" != healthy ]] || break
    preview_state=$("${preview_docker[@]}" inspect --format '{{.State.Status}}' "$preview_backend")
    [[ "$preview_state" == running ]] || { echo "Application stopped during startup." >&2; exit 1; }
    sleep 1
done
[[ "$preview_health" == healthy ]] || { echo "Application did not become healthy." >&2; exit 1; }

# nginx resolves its upstream at startup, so wait until the app stays running.
preview_proxy=$("${preview_docker[@]}" create --pull never \
    --name "nzgd-map-preview-$$-proxy" \
    --network "$preview_network" --publish '127.0.0.1::80' \
    --mount "type=bind,source=$preview_root/docker/preview-proxy.conf,target=/etc/nginx/nginx.conf,readonly" \
    --no-healthcheck --entrypoint nginx "$preview_image" -g 'daemon off;')
"${preview_docker[@]}" start "$preview_proxy" >/dev/null
# Let Docker allocate a free loopback port atomically; older previews can remain
# running. Read back the assigned endpoint for both checks and the browser URL.
preview_address=$("${preview_docker[@]}" port "$preview_proxy" 80/tcp)
[[ "$preview_address" =~ ^127\.0\.0\.1:[0-9]+$ ]] || { echo "Unexpected preview address: $preview_address" >&2; exit 1; }
preview_origin="http://$preview_address"
python3 "$preview_root/docker/check_preview.py" "$preview_origin"

if [[ "$preview_keep" == --keep ]]; then
    echo "Private preview: $preview_origin/nzgd"
    read -r -p "Press Enter when finished to remove the preview containers and network. "
fi
echo "Removing the preview containers. The local image is retained; nothing was published or deployed."
