# ADR-001 — Data Layer (Parquet + DuckDB + PostgreSQL + MinIO + DVC)
Status: Accepted
Context: Analítica completa 00–99 y trazabilidad sin comprometer reproducibilidad.
Decision:
- Parquet canónico para histórico y predicciones.
- DuckDB para consultas analíticas locales/CI.
- PostgreSQL para metadatos online (catálogo de reglas, snapshots).
- MinIO (S3) para artefactos (predicciones/diagnósticos/backtests).
- DVC para versionado de datasets/artefactos.
Consequences: Reproducible, portable; escalable a Spark/Trino cuando convenga.
