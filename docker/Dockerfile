# Specifies that Alpine Linux be used as the base image for the Docker container.
# Ubuntu Linux can also be used, but the resulting images are approximately 10 times larger.
FROM python:3.12-alpine  

# Uses Alpine’s package manager `apk` to install the necessary packages (`git`, `nginx`, 
# `sqlite` libraries, and `uwsgi` for `Python3`) without caching, which helps keep the image size small.
RUN apk add --no-cache git nginx sqlite-libs uwsgi-python3  

# Creates the `/data` directory if it does not already exist.
RUN mkdir -p /data  

# Creates the `/usr/var/nzgd_map-instance` directory, which is used for storing 
# instance-specific data for the NZGD map service.
RUN mkdir -p /usr/var/nzgd_map-instance  

# Changes the ownership of the `/usr/var/nzgd_map-instance` directory (and its contents) 
# recursively to the user and group “nobody” for improved security.
RUN chown -R nobody:nobody /usr/var/nzgd_map-instance  

# Installs the `nzgd_map` package directly from a GitHub repository without using the pip cache, 
# ensuring the latest version for container deployment.
RUN pip install --no-cache-dir "git+https://github.com/ucgmsim/nzgd_map#egg=nzgd_map" uwsgi  

# Copies the `nzgd.ini` configuration file from the build context into the container’s root directory.
COPY nzgd.ini /nzgd.ini  

# Copies the `start.sh` script from the build context into the container’s root directory.
COPY start.sh /start.sh  

# Grants execute permission to the `start.sh` script so that it can be run as a command when the container starts.
RUN chmod +x /start.sh  

# Copies the `nginx.conf` file from the build context into the container’s `/etc/nginx/` directory, 
# replacing the default nginx configuration.
COPY nginx.conf /etc/nginx/nginx.conf  

# Sets the default command to run when the container starts; it executes the `start.sh` script, 
# which starts nginx in the foreground.
CMD /start.sh  