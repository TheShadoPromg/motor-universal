# Fase 5 - Reentrenamiento OOS (activadores con Train, tuning en Valid, evaluación en Test)

Objetivo: obtener métricas 100 % out-of-sample reentrenando las fases de auditoría (2/2.5/3) y hazard (2.H/3.H) sólo con Train, y tunear β/λ en Valid antes de medir en Test.

## Resumen del flujo
1) Subset de eventos a Train_struct (configurable; e.g. 2011-10-19 a 2019-12-31 para la ventana larga).
2) Reejecuta Fase 2, 2.5 y 3 sólo con ese Train -> activadores estructurales entrenados.
3) Reejecuta Fase 2.H y 3.H sólo con Train -> activadores de hazard entrenados.
4) Tuning de β/λ en Valid (ej. 2020-01-01 a 2022-12-31) usando phase4_tune con activadores TRAIN (estructural + hazard).
5) Evaluación final en Test (ej. 2023-01-01 a fin de datos) con hiperparámetros fijos.
6) Persistencia completa en Parquet/CSV (sin truncar grids ni resultados).

## CLI
```bash
python -m engine.backtesting.phase5_oos \
  --events-db-url postgresql://admin:admin@localhost:5432/motor \  # o --events-path data/raw/eventos_numericos.csv
  --base-dir data/backtesting/oos \
  --window-id 1 \
  --train-start 2011-10-19 --train-end 2019-12-31 \
  --valid-start 2020-01-01 --valid-end 2022-12-31 \
  --test-start 2023-01-01 \
  --beta-grid 0.5,1.0,1.5,2.0 \
  --lambda-grid 0.5,0.7,0.85,1.0 \
  --include-brier \
  --verbose
```

## Qué hace internamente
- Filtra eventos a Train y los guarda en `base_dir/window_{id}/inputs/eventos_train.parquet|csv`.
- Fase 2 con Train: salida en `.../audit/estructural_train/`.
- Fase 2.5 con Train: salida en `.../audit/estructural_fase2_5_train/`.
- Fase 3 con Train: activadores en `.../activadores/activadores_dinamicos_fase3_para_motor.parquet`.
- Fase 2.H con Train: `.../audit/hazard/hazard_global_resumen.*`, `hazard_numero_resumen.*` (y opcional `hazard_opportunities.*`).
- Fase 3.H con Train: `.../activadores/hazard/activadores_hazard_para_motor.parquet|csv`.
- Llama a `phase4_tune` con activadores TRAIN estructurales + hazard y los splits Train/Valid/Test proporcionados:
  - Grid β/λ (por defecto: beta={0.5,1.0,1.5,2.0}, lambda={0.5,0.7,0.85,1.0}).
  - Selección de mejores en Valid con filtros y score compuesto.
  - Evaluación final en Train/Valid/Test con hiperparámetros óptimos.

## Salidas
- Directorio base: `data/backtesting/oos/window_{id}/`
  - `inputs/` (eventos_train)
  - `audit/estructural_train`, `audit/estructural_fase2_5_train`
  - `activadores/` (activadores entrenados)
  - Tuning y evaluación (parquet/csv):
    - `phase4_grid_valid.parquet|csv` (todas las combinaciones en Valid)
    - `best_phase4_params.parquet|csv` (ganadores por modelo)
    - `phase4_results_final.parquet|csv` (Train/Valid/Test con hiperparámetros fijos)
    - `phase4_sensitivity_test.parquet|csv` (perturbaciones ±beta, ±lambda en Test)
    - `phase4_results_segments.parquet|csv` (segmentos de Test: mitades, día de semana)

## Extensiones recomendadas
- Multi-ventana: ejecutar varias `window_id` con distintos splits (ej. Window1: P1->P2->P3, Window2: P1+P2->P3->P4) y comparar estabilidad.
- Segmentación en Test: partir Test en sub-bloques (años, día de semana) y medir lifts por segmento (exportado en `phase4_results_segments.*`).
- Sensibilidad: evaluar β/λ perturbados alrededor de los óptimos en Test (exportado en `phase4_sensitivity_test.*`).

## Limitaciones y rigor
- El reentrenamiento de reglas usa sólo Train_struct, evitando fuga de Test/Valid a nivel de reglas.
- El tuning de β/λ usa sólo Valid; Test no interviene en la selección.
- La lógica diaria sigue sin look-ahead (usa lags previos).
