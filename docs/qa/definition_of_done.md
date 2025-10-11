# Definition of Done
Fase 0: Datos gobernados
- Parquet canónico, 100% fechas, 00–99, validaciones (GE).
Fase 1: Catálogo/linaje
- rules_catalog y lineage operativos en Postgres.
Fase 2: Motores por capa
- Cross/Structural/Derived con trazas por número.
Fase 3: Fusión+Calibración
- Distribución 00–99 calibrada (∑=1) y tipo de convergencia.
Fase 4: Backtesting
- Walk-forward con logloss/brier/ece y lift por cuantiles.
Fase 5: API
- Endpoints GET listos, sin truncar; OpenAPI publicado.
Fase 6: LLM
- Explicador/crítico con guardrails (no probabilístico).
