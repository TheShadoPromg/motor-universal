# Mapa de Auditoría (Fases 1, 2, 2.5, 3) y Activadores

Este documento detalla las fases de auditoría, sus entradas, salidas, comandos y columnas clave, para uso operativo o delegación a otros agentes.

## Visión general
- **Fase 1 (randomness)**: Verifica compatibilidad con aleatoriedad básica (chi2, par/impar, alto/bajo, decenas, repetición, rachas).
- **Fase 2 (estructural)**: Sesgos origen→destino por lag/posición.
- **Fase 2.5**: Estabilidad por periodos; clasifica sesgos en core_global / periodico / extended_global.
- **Fase 3**: Traduce sesgos validados a activadores dinámicos con pesos.
- **Carga a DB**: Inserta activadores en Postgres para consumo del motor.

Entradas fundamentales:
- Histórico de sorteos `eventos_numericos` con columnas `fecha`, `posicion`, `numero` (00-99), tres filas por fecha.

Convenciones de salidas:
- Formatos estándar: Parquet (principal) + CSV cortesía.
- Directorios bajo `data/audit/` y `data/activadores/`.

## Comandos clave (CLI)

### Fase 1 - Aleatoriedad
```bash
python -m engine.audit.randomness \
  --input data/raw/eventos_numericos.csv \
  --run-date YYYY-MM-DD \
  --output-dir data/audit/randomness
```
Salidas típicas (parquet/csv):
- `frecuencia_global_*`, `frecuencia_por_posicion_*`
- `par_impar_*`, `alto_bajo_*`, `decenas_*`
- `repeticion_dias_consecutivos_*`, `condicional_reaparicion_*`
- `rachas_*`

Columnas clave (ejemplos):
- `chi2_global`, `p_value_global`, `N_sorteos`
- Para repetición: `prob_empirica_repeticion`, `p_value`
- Para rachas: `longitud_media`, `longitud_maxima`

### Fase 2 - Estructural
```bash
python -m engine.audit.estructural \
  --input data/raw/eventos_numericos.csv \
  --output-dir data/audit/estructural
```
Salida: tablas de sesgos origen→destino por lag/pos, segmentadas por periodo (ver config interna `CONFIG["segmentos_periodo"]`).

Columnas típicas:
- `tipo_relacion` (numero/consecutivo/espejo según implementación)
- `numero_base`, `numero_destino`
- `pos_origen`, `pos_destino`, `lag`
- métricas de delta relativo, z-score, flags por periodo (es_fuerte, es_debil)

### Fase 2.5 - Estabilidad por periodo
```bash
python -m engine.audit.estructural_fase2_5 \
  --input-dir data/audit/estructural \
  --output-dir data/audit/estructural_fase2_5
```
Salidas:
- `sesgos_fase2_5_core_y_periodicos.parquet|csv`
- `sesgos_fase2_5_por_periodo.parquet|csv`

Columnas clave:
- `clasificacion_fase2_5` (core_global, periodico, extended_global)
- `mean_delta_rel_periods` (magnitud del sesgo promedio)
- `stability_score` (0-1)
- Por periodo: `es_fuerte`, `es_debil`, `tiene_datos`

### Fase 3 - Activadores dinámicos
```bash
python -m engine.audit.estructural_fase3_activadores \
  --core-path data/audit/estructural_fase2_5/sesgos_fase2_5_core_y_periodicos.parquet \
  --periodos-path data/audit/estructural_fase2_5/sesgos_fase2_5_por_periodo.parquet \
  --output-dir data/activadores --format parquet
```
Salidas:
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
- `Regla_Condicional` (trazabilidad textual)

### Carga a DB de activadores
```bash
python -m engine.audit.activadores_loader \
  --input data/activadores/activadores_dinamicos_fase3_para_motor.parquet \
  --db-url postgresql://admin:admin@localhost:5432/motor \
  --table activadores_dinamicos_fase3 \
  --run-date YYYY-MM-DD \
  --if-exists replace
```

## Flujos Prefect (alternativa)
- Fase 1: `python flows/audit_pipeline.py`
- Fase 2: `python flows/audit_structural_pipeline.py`
- Fase 2.5: `python flows/audit_structural_fase2_5_pipeline.py`
- Fase 3: `python flows/audit_structural_fase3_pipeline.py`
- Carga DB: `python flows/load_activadores_pipeline.py`

## Rutas de salida estándar
- Fase 1: `data/audit/randomness/`
- Fase 2: `data/audit/estructural/`
- Fase 2.5: `data/audit/estructural_fase2_5/`
- Fase 3: `data/activadores/`

## Buenas prácticas y notas
- Formato preferido: Parquet (CSV como cortesía).
- Evitar look-ahead: cada fase usa lags; no debe consumir `t` o `t+`.
- En fases 2.5/3 se trabaja con histórico completo (hay optimismo implícito); para rigor, recalcular por ventanas train/test en escenarios futuros.
- Mantener `.env` completo para Docker/Prefect si se usan flujos.

## Ejemplo mínimo (pipeline manual completo)
```bash
# Fase 1
python -m engine.audit.randomness --input data/raw/eventos_numericos.csv --output-dir data/audit/randomness

# Fase 2
python -m engine.audit.estructural --input data/raw/eventos_numericos.csv --output-dir data/audit/estructural

# Fase 2.5
python -m engine.audit.estructural_fase2_5 --input-dir data/audit/estructural --output-dir data/audit/estructural_fase2_5

# Fase 3
python -m engine.audit.estructural_fase3_activadores \
  --core-path data/audit/estructural_fase2_5/sesgos_fase2_5_core_y_periodicos.parquet \
  --periodos-path data/audit/estructural_fase2_5/sesgos_fase2_5_por_periodo.parquet \
  --output-dir data/activadores --format parquet

# Carga a DB (opcional)
python -m engine.audit.activadores_loader \
  --input data/activadores/activadores_dinamicos_fase3_para_motor.parquet \
  --db-url postgresql://admin:admin@localhost:5432/motor \
  --table activadores_dinamicos_fase3 \
  --run-date YYYY-MM-DD --if-exists replace
```
