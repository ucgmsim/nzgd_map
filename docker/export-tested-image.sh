#!/usr/bin/env bash
# Workstation-only export of an already tested image. No build, push, or SSH.
set -euo pipefail
umask 077

case "$(hostname -s)" in
    mantle|ucquakecore2p|2p)
        echo "Run this export on the development workstation." >&2
        exit 1
        ;;
esac
if (( $# != 2 )) || [[ ! "$1" =~ ^sha256:[a-f0-9]{64}$ ]]; then
    echo "Usage: bash docker/export-tested-image.sh sha256:IMAGE_ID /absolute/output.tar.gz" >&2
    exit 2
fi
export_image="$1"
export_output="$2"
[[ "$export_output" == /* ]] || { echo "Use an absolute output path." >&2; exit 2; }
[[ ! -e "$export_output" ]] || { echo "Output already exists: $export_output" >&2; exit 1; }
mkdir -p -- "$(dirname -- "$export_output")"
export_docker=(docker --host unix:///var/run/docker.sock)
if ! "${export_docker[@]}" version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    export_docker=(sudo docker --host unix:///var/run/docker.sock)
    "${export_docker[@]}" version --format '{{.Server.Version}}'
fi
export_actual=$("${export_docker[@]}" image inspect --format '{{.Id}}' "$export_image")
[[ "$export_actual" == "$export_image" ]] || { echo "Image ID did not match." >&2; exit 1; }
"${export_docker[@]}" image inspect --format 'Exporting {{.Id}} ({{.Size}} bytes; {{.Os}}/{{.Architecture}})' "$export_image"
export_temporary=$(mktemp --tmpdir="$(dirname -- "$export_output")" .nzgd-map-image.XXXXXXXX)
trap 'rm -f -- "$export_temporary"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${export_docker[@]}" image save "$export_image" | gzip -1 > "$export_temporary"
gzip -t -- "$export_temporary"
# Link without replacing an existing output, then remove the temporary name.
ln -- "$export_temporary" "$export_output"
sha256sum -- "$export_output"
stat --printf='Archive size: %s bytes\n' -- "$export_output"
echo "Export complete. No image was built, published, or transferred."
