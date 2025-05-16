## Introduction

This repository contains the source code for the `nzgd_map` package. This is a web application
that enables access to analysis-ready data products derived from data hosted on the New Zealand Geotechnical 
Database (NZGD). This repository also contains files for building a Docker image that can be used to run the
`nzgd_map` package in a containerized environment.

## Installation

### Installing outside a Docker container

#### Installing an uneditable version

To install an uneditable version of `nzgd_map` outside a Docker container, simply enter the following 
command in a terminal:

* `pip install "git+https://github.com/ucgmsim/nzgd_map"` 

#### Installing an editable version

To install an editable version of `nzgd_map` outside a Docker container, enter the commands below in a terminal:

* Change to the directory that you want to clone the repository to
    * e.g., `cd /home/username/src/` 
* Clone the repository from GitHub
    * `git clone git@github.com:ucgmsim/nzgd_map.git`

* Install the package in editable mode
    * `cd nzgd_map`
    * `pip install -e .`

### Installing inside a Docker container

#### Build and push the image of the Docker container

These steps can be carried out on any machine that has Docker (i.e., they do not need to be carried out on the machine that will host the website).  

* Clone this repo
    * `git clone git@github.com:ucgmsim/nzgd_map.git`

* In a terminal, navigate to the `docker` folder in the cloned repo (the location will depend on where the repo was cloned)
    * e.g., `cd /home/username/nzgd_map/docker`

* Build the Docker image, which includes the `nzgd_map` source files due to a `git clone` command in `Dockerfile`. In the following command, `earthquakesuc` is our Docker Hub username.

    * `docker build -t earthquakesuc/nzgd_map .` (if any cached `nzgd_map` source files can be kept)

    * `docker build --no-cache -t earthquakesuc/nzgd_map .` (if any cached `nzgd_map` source files should be overwritten, for example if `nzgd_map` has been updated)

* Push the image to Docker Hub (see the section below on logging in to Docker Hub)
    * `docker push earthquakesuc/nzgd_map`


#### Create a dedicated user account on the host machine to run the service

This guide uses **Rootless Docker**, meaning that the Docker installation is for a specific user that does not require `sudo`. If you need to install Rootless Docker on your system, follow [this guide](https://docs.docker.com/engine/security/rootless/).


* Create a user called `nzgd_map` to run the systemd service
    *  `sudo useradd -m -s /bin/bash nzgd_map` where `-m` creates a home directory and `-s /bin/bash` sets the default shell to bash.

* Temporarily give the `nzgd_map` user `sudo` privileges to make the next few steps simpler (`sudo` privileges will be revoked later)
    *  `sudo usermod -aG sudo nzgd_map`

* Change to the `nzgd_map` user
    * e.g., `exit`
    * e.g., `ssh nzgd_map@mantle` (if hosting on Mantle)

* Set the `nzgd_map` user's Docker to start at startup and also start it now 
    * `sudo systemctl enable --now docker` where `enable` sets Docker to start at startup and `--now` also starts it now.

* Allow the `nzgd_map` user to linger so it will be available at startup 
    * `sudo loginctl enable-linger $(whoami)`
* Add the following line to the `nzgd_map` user's `~/.bashrc` file to point to the user's Docker socket
    * `export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock`

* Change back to your main account with `sudo` privileges
    * e.g., `exit`
    * e.g., `ssh main_username@mantle`

* Revoke the `nzgd_map` user's `sudo` privileges as they are no longer needed 
    * `sudo deluser nzgd_map sudo`

#### Start the service that runs the website

* Copy the `nzgd_map.service` file to the host machine
    * e.g., `cp nzgd_map.service /etc/systemd/system/`
* Reload the systemd unit files
    *  `sudo systemctl daemon-reload`
* Set the service to start on startup with `enable`
    * `sudo systemctl enable nzgd_map.service`

* Start the service to pull and run the latest image of the container from Docker Hub
    * `sudo systemctl start nzgd_map.service` (if **not** already running)
    * `sudo systemctl restart nzgd_map.service` (if already running)

* Check the status of the service to ensure it's running
    * `sudo systemctl status nzgd_map.service` 

* Check the status of the `nzgd_map` container (as the `nzgd_map` user)
    * `docker ps`

### Logging in to Docker Hub
Open a terminal and enter the following command:
`sudo docker login`

The terminal will show a message like the following:

    USING WEB BASED LOGIN
    To sign in with credentials on the command line, use 'docker login -u <username>'

    Your one-time device confirmation code is: XXXX-XXXX
    Press ENTER to open your browser or submit your device code here: https://login.docker.com/activate

    Waiting for authentication in the browser…

If a web browser does not open automatically, copy the URL provided in the message and paste it into a 
web browser. On the web page that opens, enter the one-time device confirmation code provided in 
the message, and our organization's Docker Hub username and password to log in.

## Files for building a Docker image

The following files in the `docker` directory set up the NZGD map service in a container, and run it on startup with systemd. 

- [Service file to run the NZGD web app](docker/nzgd_map.service)
- [Dockerfile defining the NZGD map container](docker/Dockerfile)
- [uWSGI configuration for NZGD server](docker/nzgd.ini)
- [nginx config exposing server outside the container](docker/nginx.conf)
- [Entrypoint script that runs when container is executed](docker/start.sh)
