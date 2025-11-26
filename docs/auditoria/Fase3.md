# Fase 3 - Activadores Dinámicos Estructurales

## Objetivo
Convertir sesgos validados en Fase 2.5 en reglas/activadores dinámicos con pesos para el motor.

## Entrada
- Fase 2.5:
  - `sesgos_fase2_5_core_y_periodicos.parquet|csv`
  - `sesgos_fase2_5_por_periodo.parquet|csv`

## Comando
```bash
python -m engine.audit.estructural_fase3_activadores \
  --core-path data/audit/estructural_fase2_5/sesgos_fase2_5_core_y_periodicos.parquet \
  --periodos-path data/audit/estructural_fase2_5/sesgos_fase2_5_por_periodo.parquet \
  --output-dir data/activadores --format parquet
```

## Salidas
- `activadores_dinamicos_fase3_raw.parquet|csv`
- `activadores_dinamicos_fase3_para_motor.parquet|csv`
- `activadores_dinamicos_fase3_core_y_periodicos.parquet|csv`
- `activadores_fase3.sqlite` (opcional)

Columnas clave:
- `NumeroObjetivo`, `NumeroCondicionante`, `Lag`
- `PosOrigen`, `PosDestino` (ANY o 1/2/3)
- `TipoRelacion` (numero, espejo, consecutivo_+1, consecutivo_-1)
- `Clasificacion_Fase2_5` (core_global, periodico, extended_global)
- `Peso_Bruto`, `Peso_Normalizado`, `Stability_Score`
- `Regla_Condicional` (texto trazable)

## Interpretación
- Los activadores representan reglas dinámicas: si el `NumeroCondicionante` aparece en `t-Lag`, se refuerza el `NumeroObjetivo` con `Peso_Normalizado`.
- `core_global` son los activadores más fuertes/estables; `periodico` pueden aportar señal pero con mayor variabilidad.
