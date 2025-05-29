## Introduction

This repository contains the source code for the `nzgd_map` package. This is a web application
that enables access to analysis-ready data products derived from data hosted on the [New Zealand Geotechnical 
Database (NZGD)](https://identity.beca.digital/identity.beca.digital/b2c_1a_beyon_nzgd_signup_signin/oauth2/v2.0/authorize?client_id=7c5e7680-f453-486a-92cd-7738c1fb5e72&scope=https%3A%2F%2Fidentity.beca.digital%2Fbeyon%2Fconsumer%20openid%20profile%20offline_access&redirect_uri=https%3A%2F%2Fnzgd.org.nz%2F&client-request-id=a86ce44b-1aae-40f9-9c1c-d2e646a00af8&response_mode=fragment&response_type=code&x-client-SKU=msal.js.browser&x-client-VER=2.30.0&client_info=1&code_challenge=R_Z8_0hCHOUe9MyS-sZUKbQ4JY_XA2RTinLrZ8ZSBcw&code_challenge_method=S256&prompt=login&nonce=dbb0e600-95c2-42bf-bf78-312917cb6938&state=eyJpZCI6ImEwNjVhYzQzLTBlNzAtNDE5Ni05OWUzLTVkMDcwNDc5M2M4MiIsIm1ldGEiOnsiaW50ZXJhY3Rpb25UeXBlIjoicmVkaXJlY3QifX0%3D). This repository also contains files for building a Docker image that can be used to run the
`nzgd_map` package in a containerized environment.

## Setting up the web app on `Mantle`

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

If the the `nzgd_map` web app is modified, a new Docker image that contains the modified
`nzgd_map` code needs to be built and pushed to Docker Hub. This can be done with any
machine that has Docker.

[`Dockerfile`](docker/Dockerfile) contains instructions for building the image of the 
Docker container.
One of these instructions installs the latest version of the `nzgd_map` package from
GitHub. To build the Docker container image, open a terminal and navigate to the 
`docker` directory in the `nzgd_map` repo 
  * `cd /location/of/repo/docker/folder`

This Flask app uses a secret key to securely manage the session. The secret key is 
passed as a build-time argument, to avoid hard coding it in the Dockerfile.  The build 
process will try to re-use cached `nzgd_map` files by default, so if the `nzgd_map` 
package has been modified, you should build with the `--no-cache` flag:
 * `docker build --no-cache -t earthquakesuc/nzgd_map .` 
 * `docker build --build-arg SECRET_KEY="EXAMPLE" --no-cache -t earthquakesuc/nzgd_map .`

 (If you can keep cached files, remove the `--no-cache` flag from the command)

To push the newly built container image to Docker Hub, ensure you are logged in to
our Docker Hub account (earthquakesuc), and then run
  * `docker push earthquakesuc/nzgd_map`

## Files for building a Docker image

The following files in the `docker` directory set up the NZGD map service in a container, and run it on startup with systemd. 

- [Service file to run the NZGD web app](docker/nzgd_map.service)
- [Dockerfile defining the NZGD map container](docker/Dockerfile)
- [uWSGI configuration for NZGD server](docker/nzgd.ini)
- [nginx config exposing server outside the container](docker/nginx.conf)
- [Entrypoint script that runs when container is executed](docker/start.sh)
