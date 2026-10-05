# Mantle release scripts

These files run the NZGD map on Mantle as a rootless Docker container owned by
the `nzgd_map` account and started by systemd. They contain no secrets: the
session key is generated on the server and kept in
`/mnt/mantle_data/nzgd_map/runtime/container.env` (root:nzgd_map, mode 0640).

| File | Purpose |
| --- | --- |
| `install-public.sh` | Run with sudo on Mantle from a staging directory. Without arguments it installs the first release on a stopped service. With `--update` it replaces the running release: it checks the new release while the old one keeps serving, then switches over and restores the previous release automatically if startup fails. |
| `rollback-public.sh` | Copied into each release directory. Restores the previous unit and, for an update, restarts and verifies the previous image. |
| `container-service.sh` | Called by the systemd unit to prepare, run, health-check, stop and clean up the container. It runs the pinned image ID with `--pull never` and read-only mounts. |
| `config.py` | Production Flask settings, mounted read-only into the container. |
| `deployment.env` | Settings of the first release (20260924), whose unit is [`../nzgd_map.service`](../nzgd_map.service). |
| `releases/<id>/` | Each later release's `deployment.env` (release and image IDs, the previous release and image, database checksum) and systemd unit. |

The installer verifies every staged file, the database checksum and the
image ID, and runs a short network-disabled check of the database, production
settings and station file as the app's user before starting the service.

## Releasing an update

1. On the workstation, build and check the candidate with
   `docker/local-preview.sh`, then export that exact image with
   `docker/export-tested-image.sh` ([instructions](../readme.md)).
2. Add `releases/<id>/deployment.env` and `releases/<id>/nzgd_map.service`,
   copied from the previous release with the new release ID, image ID and
   release paths, and the previous release and image IDs.
3. Put the bundle in one directory: `image.tar.gz`; `install-public.sh`,
   `rollback-public.sh`, `container-service.sh` and `config.py` from this
   directory; and the release's `deployment.env` and `nzgd_map.service`. Then
   write its manifest:

   ```bash
   sha256sum image.tar.gz install-public.sh rollback-public.sh container-service.sh \
     config.py deployment.env nzgd_map.service > SHA256SUMS
   ```

4. Copy the bundle to a new directory on Mantle,
   `/mnt/mantle_data/nzgd_map-staging-<id>` (mode 0700), and run
   `sha256sum --check SHA256SUMS` there.
5. On Mantle, install it:

   ```bash
   sudo bash /mnt/mantle_data/nzgd_map-staging-<id>/install-public.sh --update
   ```

   The site is unavailable for about a minute while the new container starts.
6. Check `http://localhost:7777/nzgd/` on Mantle and the public URL in a browser.
7. To return to the previous release:

   ```bash
   sudo bash /mnt/mantle_data/nzgd_map/releases/<id>/rollback-public.sh
   ```

To replace the session key, rewrite the `SECRET_KEY` line of `container.env`
with a new random value before installing, keeping its owner and mode. The next
container start uses it, and existing browser sessions are reset.

The tests in `tests/test_mantle_service.py`, `tests/test_release_update.py` and
`tests/test_deployment_preflight.py` exercise these scripts with stand-in
commands; they need neither Docker nor sudo.
