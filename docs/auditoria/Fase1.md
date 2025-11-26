# Fase 1 - Auditoría de Aleatoriedad (randomness)

## Objetivo
Evaluar si el histórico de sorteos (3 números distintos 00-99 por día) es compatible con un proceso aleatorio uniforme. Sirve como baseline estadístico y para descartar sesgos gruesos.

## Entrada
- Archivo o tabla `eventos_numericos` con columnas:
  - `fecha` (YYYY-MM-DD)
  - `posicion` (1/2/3 o texto equivalente)
  - `numero` (00-99)

## Comando
```bash
python -m engine.audit.randomness \
  --input data/raw/eventos_numericos.csv \
  --run-date YYYY-MM-DD \
  --output-dir data/audit/randomness
```

## Salidas (parquet/CSV)
- `frecuencia_global_numeros`, `frecuencia_global_resumen`
- `frecuencia_por_posicion`, `frecuencia_por_posicion_resumen`
- `par_impar_global`, `par_impar_global_resumen`
- `par_impar_por_posicion`, `par_impar_por_posicion_resumen`
- `alto_bajo_*`, `decenas_*`
- `repeticion_dias_consecutivos`, `repeticion_dias_consecutivos_resumen`
- `condicional_reaparicion`, `condicional_reaparicion_resumen`
- `rachas_par_impar`, `rachas_par_impar_resumen`
- `rachas_alto_bajo`, `rachas_alto_bajo_resumen`
- `rachas_repeticion`, `rachas_repeticion_resumen`

Columnas clave (ejemplos):
- `chi2_global`, `p_value_global`, `N_sorteos`
- Para par/impar/alto-bajo/decenas: `chi2`, `p_value`, `desviacion_significativa_bool`
- Repetición: distribución 0-3 repetidos, `p_value`
- Condicional reaparición: `p_emp_cond`, `p_teorica`
- Rachas: número de rachas, longitudes medias y máximas

## Interpretación rápida
- p-values altos sugieren compatibilidad con aleatoriedad; p-values muy bajos podrían indicar sesgo.
- Esta fase no detecta patrones condicionales finos; solo descarta/alerta sobre sesgos gruesos.
