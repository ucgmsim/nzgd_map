#!/usr/bin/env bash
set -euo pipefail

: "${SECRET_KEY:?Set SECRET_KEY when starting the container}"
: "${NZGD_DATABASE_PATH:?Set NZGD_DATABASE_PATH when starting the container}"
if [[ ! -r "$NZGD_DATABASE_PATH" ]]; then
    echo "The configured database is not readable: $NZGD_DATABASE_PATH" >&2
    exit 1
fi

uwsgi --ini /nzgd.ini &
app_pid=$!
nginx -g 'daemon off;' &
nginx_pid=$!

cleanup() {
    trap - EXIT TERM INT
    kill -TERM "$app_pid" "$nginx_pid" 2>/dev/null || true
    wait "$app_pid" "$nginx_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Stop the container if either server exits, so nginx cannot mask a failed app.
set +e
wait -n "$app_pid" "$nginx_pid"
status=$?
if (( status == 0 )); then
    status=1
fi
exit "$status"
