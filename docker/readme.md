# Local container preview

The container runs the app with nginx and uWSGI. Both use the same Python 3.12
runtime, the app is built from this checkout, and the runtime dependencies in
`requirements.txt` are pinned, with uWSGI 2.0.31 compiled for the image's Python.

**Run the following only on a development workstation.** Server deployment is
covered by the maintainers' internal runbooks.

From the repository root:

```bash
bash docker/local-preview.sh /path/to/uc_nzgd_v0p8p2_20260709_deduped.db --keep
```

The script uses this workstation's `/var/run/docker.sock` explicitly. It prompts
for sudo in your terminal if ordinary Docker access is unavailable. It refuses
to run on the deployment hosts, builds `nzgd-map:preview-20260923`, prints its
image ID and size, and creates a temporary private Docker network with two
containers:

- The app container, with the selected database mounted read-only.
- A proxy reproducing the production NZGD path rewrite and Content-Security-Policy.

Only the proxy is published, on **127.0.0.1 with an automatically assigned port**,
so an older preview can stay running. The script reads the assigned address back
from Docker and prints the full preview URL. See Docker's
[port allocation documentation](https://docs.docker.com/get-started/docker-concepts/running-containers/publishing-ports/).
It is not exposed on external network interfaces. A temporary session key is
generated at runtime and is not written into the image. The Docker client reads
it from a private temporary environment file (mode `0600`), including when run
through sudo. The proxy starts after the app becomes healthy.

The script then runs HTTP checks against the July 2026 data: report counts, all
three availability selections, the proxied `/nzgd` path, matching JavaScript,
CSS, compressed map responses, separate CPT/SPT reports, CSV downloads, live
query validation, requests carrying several KB of extra headers, GeoNet
upload and reset, and session redirects. With `--keep`, it leaves the preview
at the printed **Private preview** URL for browser checks; press Enter in the
terminal when finished. The script removes its containers, network, and
temporary config; the image and build cache remain locally. Omit `--keep` to
remove the preview immediately after the checks. Containers are created before
they are started so cleanup also removes a container whose startup fails; only
containers from this invocation are removed.

To reuse an image that has already been built, add `--image` with its full
`sha256:` image ID. This skips the build and runs both containers from that exact
local image without pulling anything.

Production TLS and the public-edge infrastructure are not reproduced by this
local preview.

## Exporting a tested image

After the preview passes, export that exact image without rebuilding it:

```bash
bash docker/export-tested-image.sh sha256:IMAGE_ID /absolute/path/image.tar.gz
```

The script verifies the image ID, writes a gzip-compressed `docker image save`
archive, tests it, and prints its SHA-256. It does not build, publish, or
transfer anything.

## Container configuration

- `Dockerfile` uses a builder stage to compile uWSGI against the same Python as
  the final image. Compiler packages are excluded from the final stage.
- `.dockerignore` at the repository root limits the build context to required
  application and build files. Databases, local instance config, credentials,
  virtual environments, editor settings, and git history are excluded.
- `nzgd.ini` runs five workers, handles the `/nzgd` prefix, and raises uWSGI's
  request buffer so proxied requests with many headers are accepted.
- `start.sh` stops the container if nginx or uWSGI exits, instead of leaving
  nginx running with a failed app. Tini forwards termination to the supervisor.
- The health check requests `/nzgd/query_help`, exercising both the WSGI app and
  the configured database without generating the full map.
- `nginx.conf` compresses map responses and normalizes the repeated slash that
  the upstream proxy configuration introduces.
- `config.py.example` scopes cookies to `nzgd_session` and `/nzgd`, using secure
  cookies for the production HTTPS URL. The HTTP preview uses a temporary
  configuration without the secure-cookie flag.
- `SECRET_KEY` must be supplied when the container starts; it is never a build
  argument or image environment variable.

The base image tag and OS packages can change on a rebuild. Record the resulting
image ID and use that exact tested image for deployment. Python dependencies are
locked, but the build is not claimed to be bit-for-bit reproducible from a moving
base tag.

To regenerate the runtime dependency lock intentionally, then repeat the app
and container checks:

```bash
uv pip compile pyproject.toml docker/requirements.in \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file docker/requirements.txt
```
