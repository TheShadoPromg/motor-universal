# Fase 2 - Auditoría Estructural

## Objetivo
Detectar sesgos origen→destino por lags en los sorteos (número que apareció en pos/origen, reaparece en pos/destino tras cierto lag).

## Entrada
- `eventos_numericos` con `fecha`, `posicion`, `numero` (00-99).

## Comando
```bash
python -m engine.audit.estructural \
  --input data/raw/eventos_numericos.csv \
  --output-dir data/audit/estructural
```

## Salidas
- Parquet/CSV en `data/audit/estructural/` con sesgos calculados por:
  - `tipo_relacion` (según implementación; típicamente numero, espejo, consecutivo +/-1)
  - `numero_base`, `numero_destino`
  - `pos_origen`, `pos_destino`
  - `lag`
  - métricas: delta relativo, z-score, flags por periodo (`es_fuerte`, `es_debil`)

## Periodos
- Usa `CONFIG["segmentos_periodo"]` (por defecto: 2011-2014, 2015-2018, 2019-2022, 2023-2025).

## Interpretación
- Sesgos fuertes (delta_rel alto y significativo) son candidatos a ser marcados como core/periodico en Fase 2.5.
- Esta fase no produce activadores; solo cuantifica relaciones origen→destino y su fuerza.
