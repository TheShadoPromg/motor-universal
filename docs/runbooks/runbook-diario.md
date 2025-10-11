# Runbook Diario
1) Ingesta (nueva fecha) → validación datos → normalización 00–99.
2) Motores: cross / structural / derived_dynamic.
3) Fusión+calibración → distribución 00–99 (sum=1).
4) Diagnóstico: logloss, brier, ece, DAP, ES, DI y clase de día.
5) Publicar artifacts: predicciones_{YYYY-MM-DD}.parquet + diagnostico_{YYYY-MM-DD}.json
6) Registrar linaje (inputs+commit+responsable).
