## Introduction

This repository contains the source code for the `nzgd_map` package. This is a web application
that enables access to analysis-ready data products derived from data hosted on the [New Zealand Geotechnical 
Database (NZGD)](https://nzgd.org.nz/). This repository also contains files for building a Docker image that can be used to run the
`nzgd_map` package in a containerized environment.

## Local development

The app supports the July 2026 deduplicated CPT/SPT database. See
[the schema migration notes](docs/schema-migration.md) for schema mappings,
report and download behaviour, and validation results.

Use Python 3.12 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export NZGD_DATABASE_PATH=/path/to/uc_nzgd_v0p8p2_20260709_deduped.db
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
flask --app nzgd_map:create_app run --host 127.0.0.1
```

Open <http://127.0.0.1:5000>. The app opens the database in read-only mode; the
database can remain outside the checkout. Keep the same secret between runs if
you want to retain browser sessions.

`NZGD_LAST_RETRIEVAL_DATE` and `NZGD_GEONET_STATIONS_PATH` are optional. Without
them or the corresponding instance files, the app shows an unknown retrieval
date and allows you to upload a station overlay. Station files contain longitude,
latitude, and station name, separated by whitespace or commas. Application
configuration can also be supplied in `instance/config.py`; see the migration
notes for keys and precedence.

Run the checks with:

```bash
python -m pytest --cov=nzgd_map --cov-report=term-missing --cov-fail-under=95 tests
ruff check nzgd_map tests
ruff format --check nzgd_map tests
```

The tests create small synthetic databases using the current schema; they do not
need access to the full NZGD database. The map defaults to all reports with
measurements. Its availability selector can restrict results to reports with or
without a valid estimate for the chosen correlations. Each record page preserves
separate reports, including individual profiles and CSV downloads.

## Container image

The [Dockerfile](docker/Dockerfile) builds the code in this checkout, from the
**repository root**, into an image that runs nginx and uWSGI with one Python
runtime. `SECRET_KEY` must be supplied when the container starts; it is never
part of the image. The [local container preview](docker/readme.md) builds a
candidate image, checks it against a copy of the production proxy route, and
exports the exact tested image for deployment.

| File | Purpose |
| --- | --- |
| [`docker/Dockerfile`](docker/Dockerfile) | Builds the image from this checkout |
| [`docker/nzgd.ini`](docker/nzgd.ini) | uWSGI configuration |
| [`docker/nginx.conf`](docker/nginx.conf) | nginx configuration inside the container |
| [`docker/start.sh`](docker/start.sh) | Starts nginx and uWSGI and stops the container if either exits |
| [`docker/local-preview.sh`](docker/local-preview.sh) | Builds and checks a candidate image on a workstation |
| [`docker/export-tested-image.sh`](docker/export-tested-image.sh) | Exports a tested image for transfer to the server |

## Deployment

The public site runs on Mantle, a QuakeCoRE server, as a rootless Docker
container started by systemd under the `nzgd_map` account. Each release installs
an exact tested image that has been copied to the server, rather than pulling
one from a registry, and the previous release can be restored with one command.
The installer, rollback script, service helper, production settings and release
files are in [`docker/mantle`](docker/mantle/README.md), which also describes the
release procedure. Changes to the servers need the maintainers' approval.
