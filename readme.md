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
export NZGD_DATABASE_PATH=/home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p8p2_20260709_deduped.db
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

## Deployment notes — review before use

Runbooks for the QuakeCoRE servers are maintained internally. Source changes
reach the live site only as a new tested image installed through an approved
service update. The [workstation preview](docker/readme.md) builds and checks
a candidate image locally.

The setup commands below are historical reference. They predate the current
database layout and service helper; **use the internal runbook instead**.

## Historical setup of the web app on `Mantle`

### Creating a user account

`Mantle` is a local server that runs most of our web applications. We will create a 
user account on `Mantle` called `nzgd_map` that will run the `nzgd_map` service:

 * `sudo useradd -m -s /bin/bash nzgd_map` 
 where `-m` creates a home directory, and `-s /bin/bash` sets bash as the default shell
 * `sudo passwd nzgd_map` to set a password

We will use `rootless docker` for this set up. If you need to install `rootless docker` 
on your system, follow [this guide](https://docs.docker.com/engine/security/rootless/). 
Otherwise, continue with the next step.

Access the `nzgd_map` user's shell with
  * `sudo machinectl shell nzgd_map@`

Now as the `nzgd_map` user, run
  * `dockerd-rootless-setuptool.sh install`

Add `export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock` to the `nzgd_map` 
user's `~/.bashrc` file to point to the Docker socket:
  * `echo 'export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock' >> ~/.bashrc`
* `source ~/.bashrc` (to reload the shell)

And finally, start `docker` (`--now`) and set it to automatically start
when the `nzgd_map` user logs in (`enable`)
  * `systemctl --user enable --now docker`

Now we will log in to `Docker Hub` so we can `pull` the Docker container image 
containing this web application. 

### Logging in to Docker Hub
To start the log in process

`docker login`

The terminal will show a message like the following:

    USING WEB BASED LOGIN
    To sign in with credentials on the command line, use 'docker login -u <username>'

    Your one-time device confirmation code is: XXXX-XXXX
    Press ENTER to open your browser or submit your device code here: https://login.docker.com/activate

    Waiting for authentication in the browser…

If a web browser does not open automatically, copy the URL provided in the message and paste it into a 
web browser. On the web page that opens, enter the one-time device confirmation code provided in 
the message, and our organization's Docker Hub username and password to log in. After logging in to `Docker Hub` as the `nzgd_map` user, exit and return to your usual account by running
  * `exit`

### Web application set up

The `nzgd_map` web application uses a database that is mounted to the Docker 
container when it starts. Create a directory on `Mantle` for this database

 * `sudo mkdir /mnt/mantle_data/nzgd_map`

Then populate it with the files in [this Dropbox folder](https://www.dropbox.com/scl/fo/qccnazln9nssgj2wpoayy/AInH1-rgxPRmw7CamBWS_mo?rlkey=vx0ru18thziqp1xgieetv39oq&st=k4sl3w7p&dl=0)


Give ownership and read permission of this folder to the `nzgd_map` user

  * `sudo chown -R nzgd_map:nzgd_map /mnt/mantle_data/nzgd_map`
  * `sudo chmod -R u+rX /mnt/mantle_data/nzgd_map`

Now copy [nzgd_map.service](docker/nzgd_map.service) to `/etc/systemd/system`. For
example, use `nano` to create a file called `nzgd_map.service` in this location
and manually paste in the file contents
  * `sudo nano nzgd_map.service`
  *  manually copy and paste in the file contents
  *  save and exit 

Get the `nzgd_map` user's User ID (UID)
  * `id -u nzgd_map`

Ensure that this UID is in the place of 1010 in the following line of 
[nzgd_map.service](docker/nzgd_map.service):
  * `Environment="DOCKER_HOST=unix:///run/user/1010/docker.sock"`

Now have `systemd` load the new unit file
  * `sudo systemctl daemon-reload`

And set the service to automatically start at start up
  * `sudo systemctl enable nzgd_map.service`

The `nzgd_map` user's Docker socket will normally only be available for running the 
container if the `nzgd_map` user is logged in. However, we can keep `nzgd_map`'s docker
socket active even if the `nzgd_map` user is not logged in by enabling `linger` 
for the `nzgd_map` user
  * `sudo loginctl enable-linger nzgd_map`

Finally, to start the service, and make the web app publicly available
  * `cd /etc/systemd/system`
  * `sudo systemctl start nzgd_map.service`

## Modifying the `nzgd_map` web app

The candidate [Dockerfile](docker/Dockerfile) now builds the code in this checkout
from the **repository root**, using nginx and uWSGI with one matching Python
runtime. It no longer installs a moving GitHub branch or accepts a session
secret at build time. `SECRET_KEY` must be provided when the container starts.

Use the [local container preview instructions](docker/readme.md) to build and
test the candidate on the development workstation, including a replica of 2p's
proxy route. The preview prints the image ID and size, mounts the database
read-only, and exposes only a loopback port. This does not publish an image or
change Mantle or 2p. Agree the image distribution and server changes separately
after the preview passes.

## Files for building a Docker image

The following files in the `docker` directory set up the NZGD map service in a container, and run it on startup with systemd. 

- [Service file to run the NZGD web app](docker/nzgd_map.service)
- [Dockerfile defining the NZGD map container](docker/Dockerfile)
- [uWSGI configuration for NZGD server](docker/nzgd.ini)
- [nginx config exposing server outside the container](docker/nginx.conf)
- [Entrypoint script that runs when container is executed](docker/start.sh)
