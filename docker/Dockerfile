# Build from the repository root: docker build -f docker/Dockerfile .
# Both stages use the same Python runtime. uWSGI's Python support is compiled
# into its wheel; no distribution Python plugin or site-packages override is used.
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    UWSGI_BUILD_CORES=2
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY docker/requirements.txt /src/docker/requirements.txt
RUN python -m pip wheel --wheel-dir /wheels -r docker/requirements.txt
COPY pyproject.toml readme.md /src/
COPY nzgd_map /src/nzgd_map
RUN python -m pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NZGD_DATABASE_PATH=/data/nzgd.db
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx libpcre2-8-0 tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels nzgd_map uwsgi \
    && rm -rf /wheels \
    && mkdir -p /data /usr/local/var/nzgd_map-instance \
    && chown nobody:www-data /usr/local/var/nzgd_map-instance

COPY docker/nzgd.ini /nzgd.ini
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY --chmod=755 docker/start.sh /start.sh

# SECRET_KEY is required at runtime; it is never a build argument or image ENV.
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1/nzgd/query_help', timeout=5).close()"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/start.sh"]
