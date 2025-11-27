# Mapa profundo de auditoría y activadores (estructural + hazard)

Ruta rápida de qué hace cada fase, dónde vive el código y qué archivos produce. Todo en Parquet con CSV opcional para trazabilidad.

## Fase 1 – Aleatoriedad
- Código: `engine/audit/randomness.py`
- Entrada: `data/raw/eventos_numericos.*` o DB.
- Salidas (ej.): `docs/auditoria/Fase1_Aleatoriedad.md` + métricas de chi2, runs, gaps, etc.

## Fase 2 – Auditoría estructural (transiciones)
- Código: `engine/audit/estructural.py`
- Entrada: eventos.
- Salidas: tablas de transiciones por tipo (`transiciones_numero_*.parquet|csv`, `transiciones_espejo_*`, `transiciones_consecutivo_*`, etc.).

## Fase 2.5 – Filtrado y clasificación de sesgos
- Código: `engine/audit/estructural_fase2_5.py`
- Entrada: outputs de Fase 2.
- Salidas clave:
  - `sesgos_fase2_5_resumen.parquet|csv`
  - `sesgos_fase2_5_por_periodo.parquet|csv`
  - `sesgos_fase2_5_core_y_periodicos.parquet|csv`

## Fase 3 – Activadores dinámicos (estructural)
- Código: `engine/audit/estructural_fase3_activadores.py`
- Entrada: sesgos Fase 2.5.
- Salidas:
  - `activadores_dinamicos_fase3_raw.parquet|csv`
  - `activadores_dinamicos_fase3_para_motor.parquet|csv`
  - (opcional) `activadores_dinamicos_fase3_core_y_periodicos.parquet|csv`

## Fase 2.H – Hazard/Recencia
- Código: `engine/audit/hazard_recencia.py`
- Entrada: eventos (Train).
- Salidas:
  - `hazard_global_resumen.parquet|csv`
  - `hazard_numero_resumen.parquet|csv`
  - (opcional) `hazard_opportunities.parquet|csv` con todas las oportunidades/hits.
- Documentación: `docs/auditoria/Fase2H_Hazard.md`.

## Fase 3.H – Activadores de Hazard
- Código: `engine/audit/hazard_activadores.py`
- Entrada: resúmenes de Fase 2.H.
- Salidas:
  - `activadores_hazard_global.parquet|csv`
  - `activadores_hazard_numero.parquet|csv`
  - `activadores_hazard_para_motor.parquet|csv`
- Documentación: `docs/auditoria/Fase3H_Activadores.md`.

## Fase 4 – Evaluador histórico
- Código: `engine/backtesting/phase4.py`
- Modelos: A_uniforme, B_core, C_core_periodico, H_hazard, H_hazard_struct (struct + hazard).
- Salidas: `phase4_summary.*`, `phase4_details.*` en `data/outputs/phase4/`.
- Documentación: `docs/backtesting/Fase4.md`.

## Fase 4.x – Tuning (beta, lambda)
- Código: `engine/backtesting/phase4_tune.py`
- Splits: Train/Valid/Test configurables; grid de beta/lambda.
- Salidas: `phase4_grid_valid.*`, `best_phase4_params.*`, `phase4_results_final.*`, `phase4_sensitivity_test.*`, `phase4_results_segments.*` (por defecto en `data/backtesting/` o en el `base_dir` de Fase 5).

## Fase 5 – OOS (reentrenamiento y tuning sin fuga)
- Código: `engine/backtesting/phase5_oos.py`
- Flujo:
  - Recorta eventos a Train/Valid/Test.
  - Reejecuta Fase 2/2.5/3 y Fase 2.H/3.H sólo con Train.
  - Llama a `phase4_tune` con activadores entrenados (estructural + hazard).
- Salidas por ventana: `data/backtesting/oos/window_{id}/...` con todas las tablas intermedias y resultados de backtesting.
- Documentación: `docs/backtesting/Fase5.md`.

## Estándares de formato
- Siempre Parquet; CSV opcional duplicado para inspección rápida.
- Columnas críticas preservan tipos numéricos (int/float) y fechas ISO.
- No se trunca nada: los grids y resúmenes guardan todas las combinaciones evaluadas.
