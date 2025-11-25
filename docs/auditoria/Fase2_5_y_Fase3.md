## Fase 2.5 – Estabilidad por período

- Entrada: `data/audit/estructural` (o `--input-dir` / `AUDIT_ESTRUCTURAL_DIR`), consume las tablas de transiciones de Fase 2.
- Salida (por defecto en `data/audit/estructural_fase2_5`):
  - Parquet (estándar): `sesgos_fase2_5_resumen.parquet`, `sesgos_fase2_5_por_periodo.parquet`, `sesgos_fase2_5_core_y_periodicos.parquet`.
  - CSV (cortesía) con los mismos nombres.
- CLI: `python -m engine.audit.estructural_fase2_5 --input-dir ... --output-dir ...`
- Formato: parquet por defecto; los CSV persisten para inspección manual.
- Flujo Prefect: (opcional) puede añadirse siguiendo el patrón de Fase 2; variables `AUDIT_ESTRUCTURAL_DIR` para entrada y `AUDIT_ESTRUCTURAL_FASE2_5_DIR` para salida.

## Fase 3 – Activadores estructurales dinámicos

- Entrada: salidas de Fase 2.5 (se prefieren los Parquet; se aceptan CSV si se indican).
- Salida (por defecto en `data/activadores`):
  - Parquet: `activadores_dinamicos_fase3_raw.parquet`, `activadores_dinamicos_fase3_para_motor.parquet`, `activadores_dinamicos_fase3_core_y_periodicos.parquet`.
  - CSV (cortesía) si `--format csv|both`.
- CLI:
  ```bash
  python -m engine.audit.estructural_fase3_activadores \
    --core-path data/audit/estructural_fase2_5/sesgos_fase2_5_core_y_periodicos.parquet \
    --periodos-path data/audit/estructural_fase2_5/sesgos_fase2_5_por_periodo.parquet \
    --output-dir data/activadores \
    --format parquet   # csv|both opcionales
  ```
- Convenciones:
  - Parquet es el formato estándar para pipelines y carga en motores/DB.
  - CSV se mantiene solo para revisión manual.
  - Pesos: `Peso_Bruto` (log1p(delta) * estabilidad * peso por clase) y `Peso_Normalizado` (media=1).
  - `Regla_Condicional` proporciona trazabilidad en texto plano.
- Flujo Prefect: `flows/audit_structural_fase3_pipeline.py` usa `AUDIT_ESTRUCTURAL_FASE2_5_DIR` para entrada y `ACTIVADORES_DIR` para la salida; ejecuta con `--format parquet`.
- Carga a base de datos: `engine/audit/activadores_loader.py` carga Parquet/CSV al esquema relacional; flujo `flows/load_activadores_pipeline.py` usa `DATABASE_URL`/`DB_URL` y `ACTIVADORES_DIR` para automatizar la carga. DDL de referencia en `ops/sql/010_create_activadores_fase3_table.sql`.
