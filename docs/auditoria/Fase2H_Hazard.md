# Fase 2.H – Hazard / Recencia (Oportunidades vs Hits)

Objetivo: detectar patrones de **recencia** (días desde la última aparición) que alteran de forma sistemática la probabilidad de que un número salga hoy. Trabaja sólo con el histórico de entrenamiento (Train) para mantener el 100 % OOS.

## Entradas
- `eventos_numericos` (CSV/Parquet o DB) con columnas:
  - `fecha` (YYYY-MM-DD)
  - `posicion` (1/2/3)
  - `numero` (00–99)
- Parámetros clave (CLI):
  - `--bins`: bins de recencia, por defecto `1-5,6-10,11-20,21-30,31-45,46-60,61-90` (si quieres 90+ debes pasar bins custom).
  - `--max-recencia`: recorte superior (clip, por defecto 90) usado también para etiquetar los bins.
  - Umbrales globales: `--min-opp-global` (3000), `--delta-global-strong` (0.10), `--delta-global-ext` (0.05), `--alpha-global` (0.01).
  - Umbrales por número: `--min-opp-num` (300), `--delta-num-strong` (0.50), `--delta-num-min` (0.30), `--alpha-num` (0.005).
  - Estabilidad: `--subwindows` (por defecto 3 subventanas internas en Train), `--min-opp-subwindow` (200) y `--alpha-subwindow` (por defecto 0.05).
  - `--include-opportunities`: guarda el dataset completo de oportunidades/hits.

## Lógica (resumen técnico)
1. Recorre los sorteos en orden, manteniendo `last_seen[numero]` para calcular `recencia = días_desde_última_salida`.
2. Construye un dataset de **oportunidades**: una fila por (fecha, número) con `recencia_bin` y `hit` (1 si el número salió ese día).
3. Estadísticos por **bin global**:
   - `n_oportunidades`, `n_hits`, `h_hat = hits / oportunidades`.
   - `delta_rel = (h_hat - 0.03) / 0.03` (0.03 = prob. teórica).
   - `z` y `p_val` (z-test aprox. normal).
4. Estadísticos por **(número, bin)** con los mismos campos.
5. **Estabilidad**: repite métricas por subventana y calcula:
   - `stability_score` = proporción de subventanas con señal positiva y p_val aceptable.
   - `signos_consistentes` = True si no hay subventanas con delta claramente negativo.
6. **Clasificación**:
   - Global: `hazard_core_global`, `hazard_extended_global`, `hazard_periodico`, `ninguno` según umbrales y estabilidad.
   - Número: `hazard_numero_core`, `hazard_numero_periodico`, `ninguno` (umbral más estricto).

## Salidas
- `hazard_global_resumen.parquet|csv`
  - `recencia_bin`, `recencia_min`, `recencia_max`
  - `n_oportunidades_total`, `n_hits_total`, `h_hat`, `delta_rel`, `z`, `p_val`
- Subventanas: `delta_rel_sub_*`, `p_val_sub_*`, `n_opp_sub_*` más columnas explícitas `delta_rel_{sub}`, `p_val_{sub}`, `oportunidades_{sub}` para cada subventana con datos.
  - `stability_score_global`, `signos_consistentes_global`, `clasificacion_hazard`
- `hazard_numero_resumen.parquet|csv`
  - `numero`, `recencia_bin`, `recencia_min`, `recencia_max`
  - `n_oportunidades_numero`, `n_hits_numero`, `h_hat_numero`, `delta_rel_numero`, `z_numero`, `p_val_numero`
- Subventanas análogas: `delta_rel_numero_sub_*`, `p_val_numero_sub_*`, `n_opp_numero_sub_*` más columnas explícitas por subventana con datos.
  - `stability_score_numero`, `signos_consistentes_numero`, `clasificacion_hazard_numero`
- (Opcional) `hazard_opportunities.parquet|csv`: dataset completo de oportunidades/hits para trazabilidad.

## CLI de referencia
```bash
python -m engine.audit.hazard_recencia \
  --input data/raw/eventos_numericos.csv \
  --output-dir data/audit/hazard_train \
  --start-date 2011-10-19 --end-date 2019-12-31 \
  --bins 1-5,6-10,11-20,21-30,31-45,46-60,61-90 \
  --min-opp-global 3000 --delta-global-strong 0.10 --delta-global-ext 0.05 --alpha-global 0.01 \
  --min-opp-num 300 --delta-num-strong 0.50 --delta-num-min 0.30 --alpha-num 0.005 \
  --subwindows 2011-10-19:2014-12-31;2015-01-01:2018-12-31;2019-01-01:2019-12-31 \
  --min-opp-subwindow 200 --alpha-subwindow 0.05 \
  --include-opportunities
```

## Notas de rigor
- Trabaja sólo con el rango Train para evitar fuga a Valid/Test.
- No trunca resultados: todos los bins y (número, bin) con datos quedan en los Parquet/CSV, aunque la clasificación sea `ninguno`.
- Usa solo dependencias estándar (math.erfc para p-val) para no romper entornos sin SciPy.
