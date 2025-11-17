FROM prefecthq/prefect:2-python3.11

WORKDIR /opt/motor

COPY ops/docker/worker-requirements.txt /tmp/worker-requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/worker-requirements.txt

# Prefect command configured via docker compose
