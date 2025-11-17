FROM python:3.11-slim

WORKDIR /opt/motor

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY apps/streamlit/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /opt/motor

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "apps/streamlit/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
