# Fase 4 - Evaluación predictiva de activadores estructurales

Objetivo: medir de forma rigurosa el poder predictivo de los activadores Fase 3 contra un baseline uniforme, sin recalcular reglas ni usar look-ahead.

## Alcance
- Modelos evaluados:
  - A_uniforme: distribución 1/100.
  - B_core: solo `core_global`.
  - C_core_periodico: `core_global + periodico`.
  - Mezcla con uniforme controlada por `lambda`.
- Métricas: Hit@K (K=5/10/15/20 por defecto), rank promedio de los 3 ganadores, log-loss, Brier opcional, lifts vs uniforme.
- Sin descubrimiento de reglas; consume activadores ya generados.

## Entradas esperadas
- `eventos_numericos` (CSV o DB):
  - `fecha` (YYYY-MM-DD)
  - `posicion` (1,2,3 o texto equivalente)
  - `numero` (00-99)
- Activadores Fase 3 (`activadores_dinamicos_fase3_para_motor` en parquet/csv o tabla DB):
  - `NumeroObjetivo`, `NumeroCondicionante`, `Lag`, `Peso_Normalizado`, `Clasificacion_Fase2_5`, `PosOrigen` (ANY o 1/2/3).

## Uso del CLI
```
python -m engine.backtesting.phase4 \
  --events-db-url postgresql://admin:admin@localhost:5432/motor \  # o --events-path data/raw/eventos_numericos.csv
  --activadores-db-url postgresql://admin:admin@localhost:5432/motor \  # o --activadores-path data/activadores/activadores_dinamicos_fase3_para_motor.parquet
  --period P1 2011-01-01 2014-12-31 \
  --period P2 2015-01-01 2018-12-31 \
  --period P3 2019-01-01 2022-12-31 \
  --period P4 2023-01-01 2025-06-04 \
  --grid-beta 0.5,1,1.5,2 \
  --grid-lambda 0.7,0.85,1 \
  --include-brier \
  --output-format both \
  --output-dir data/outputs/phase4 \
  --output-db-url postgresql://admin:admin@localhost:5432/motor \
  --output-db-table backtest_phase4_summary
```

Parámetros clave:
- `--beta-core/--beta-full`: temperatura del softmax (si no se usa grid).
- `--lambda-core/--lambda-full`: mezcla con uniforme (1.0 = no mezcla).
- `--grid-beta/--grid-lambda`: listas para grid-search; elige la combinación con menor log-loss por modelo/periodo.
- `--period LABEL START END`: bloques opcionales; si se omite se usa todo el rango.

## Salidas
- Directorio `data/outputs/phase4/`:
  - `phase4_summary.parquet|csv`: una fila por periodo y modelo con HR@K, lifts, RankPromedio, LogLoss, Brier, Beta, Lambda.
  - `phase4_details.parquet|csv`: detalle diario por modelo (fecha, hits@K, rank/prom, logloss_sum, brier, prob/rank de los 3 ganadores).
- Base de datos (opcional, si se pasa `--output-db-url`):
  - Tabla `backtest_phase4_summary` (resumen).
  - Tabla `backtest_phase4_summary_details` (detalle diario).

## Fase 4.x - Tuning de hiperparámetros (beta, lambda)
- CLI: `python -m engine.backtesting.phase4_tune --events-db-url ... --activadores-db-url ...`
- Splits por defecto:
  - Train = P1+P2 (2011-10-19 a 2018-12-31)
  - Valid = P3 (2019-01-01 a 2022-12-31)
  - Test  = P4 (2023-01-01 a fin de datos)
- Grid por defecto:
  - beta: 0.5, 1.0, 1.5, 2.0
  - lambda: 0.5, 0.7, 0.85, 1.0
- Selección: mejor combinación en Valid (por modelo) con filtros mínimos y score compuesto; evalúa Train/Valid/Test con esos hiperparámetros.
- Salidas en `data/backtesting/`:
  - `phase4_grid_valid.parquet|csv` (todas las combinaciones)
  - `best_phase4_params.parquet|csv` (ganadores por modelo)
  - `phase4_results_final.parquet|csv` (Train/Valid/Test con hiperparámetros óptimos)
  - `phase4_sensitivity_test.parquet|csv` (perturbaciones ±beta, ±lambda en Test)
  - `phase4_results_segments.parquet|csv` (segmentos de Test: mitades y día de semana)

## Consideraciones metodológicas
- No hay look-ahead: cada día usa solo lags definidos en los activadores.
- Si ningún activador aplica en un día, el modelo recurre a uniforme (o mezcla si `lambda<1`).
- El grid se selecciona por menor log-loss; ajustar si se prefiere otra métrica.
- Para rigor pleno se recomienda, en una fase posterior, recalcular activadores por ventana de entrenamiento y evaluar en test separado.
