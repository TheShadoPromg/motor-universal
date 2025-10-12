FROM ghcr.io/mlflow/mlflow:latest
# Añade el driver de Postgres para SQLAlchemy
RUN pip install --no-cache-dir psycopg2-binary==2.9.9
