# ADR-002 — Orquestación y API
Status: Accepted
Decision:
- Prefect para jobs (ingesta→motores→fusión→calibración→artefactos).
- FastAPI para servicio REST: /predict/{fecha}, /diagnostics/{fecha},
  /explain/{fecha}/{numero}, /backtest/summary (sin truncar distribución).
- OpenAPI 3.0 en /api/contracts/openapi.yaml.
