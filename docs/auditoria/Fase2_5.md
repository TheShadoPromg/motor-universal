# Fase 2.5 - Estabilidad por periodo

## Objetivo
Evaluar la estabilidad temporal de los sesgos estructurales (Fase 2) y clasificarlos:
- `core_global`: fuertes y consistentes entre periodos.
- `periodico`: presentes con variación por periodo.
- `extended_global`: (opcional) menos fuertes pero globales.

## Entrada
- Salidas de Fase 2 (`data/audit/estructural/`):
  - Sesgos origen→destino por lag/pos con métricas por periodo.

## Comando
```bash
python -m engine.audit.estructural_fase2_5 \
  --input-dir data/audit/estructural \
  --output-dir data/audit/estructural_fase2_5
```

## Salidas
- `sesgos_fase2_5_core_y_periodicos.parquet|csv`
- `sesgos_fase2_5_por_periodo.parquet|csv`

Columnas clave:
- `clasificacion_fase2_5` (core_global, periodico, extended_global)
- `mean_delta_rel_periods` (magnitud promedio del sesgo)
- `stability_score` (0-1)
- Por periodo: `es_fuerte`, `es_debil`, `tiene_datos`

## Interpretación
- Solo los sesgos clasificados como core_global/periodico pasan a Fase 3 (activadores).
- stability_score alto + mean_delta_rel alto sugieren señales robustas.
