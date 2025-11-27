# Fase 3.H – Conversión de patrones de Hazard a activadores

Objetivo: traducir los patrones de recencia detectados en Fase 2.H en activadores que el motor pueda consumir (parquet/csv y listo para la capa de backtesting).

## Entradas
- `hazard_global_resumen.parquet|csv`
- `hazard_numero_resumen.parquet|csv`
- Configuración principal (CLI de `engine.audit.hazard_activadores`):
  - `--input-dir`: carpeta con los resúmenes de Fase 2.H.
  - `--output-dir`: carpeta destino.
  - `--format`: `parquet` (default) o `csv` adicional.

## Reglas de selección
- Global:
  - Incluye bins con `clasificacion_hazard` en `{hazard_core_global, hazard_extended_global, hazard_periodico}`.
- Por número:
  - Incluye filas con `clasificacion_hazard_numero` en `{hazard_numero_core, hazard_numero_periodico}`.
- No se descarta nada por peso: si pasa la clasificación, se convierte en activador.

## Cálculo de pesos
Para cada patrón (global o número) se calcula:
- `class_factor`:
  - 1.5 para `core`
  - 1.2 para `extended`
  - 1.0 para `periodico`
- `mag_factor = log1p(max(delta_rel_medio, 0))`
- `stab_factor = 0.5 + 0.5 * stability_score` (mapea [0,1] a [0.5,1.0])
- `Peso_Bruto = class_factor * mag_factor * stab_factor`
- `Peso_Normalizado = Peso_Bruto / mean(Peso_Bruto_{>0})` (media 1.0 sobre activadores con peso>0), calculado **de forma conjunta** sobre todos los activadores hazard (global + número) para mantener la escala relativa entre ambos grupos.

## Salidas
- `activadores_hazard_global.parquet|csv`
  - Una fila por bin global activo:
    - `NumeroObjetivo` = 00–99 (uno por número), `RecenciaBin`, `RecenciaMin`, `RecenciaMax`
    - `TipoPatron = hazard_global`
    - `Clasificacion_Hazard`, `Peso_Bruto`, `Peso_Normalizado`
    - `Regla_Condicional` trazable
- `activadores_hazard_numero.parquet|csv`
  - Una fila por (numero, bin) activo con los mismos campos más `Clasificacion_Hazard_Numero`.
- `activadores_hazard_para_motor.parquet|csv`
  - Unión de global + número con columnas listas para el motor/backtesting.

## Ejemplo de fila (global)
| NumeroObjetivo | RecenciaBin | RecenciaMin | RecenciaMax | TipoPatron    | Clasificacion_Hazard | Peso_Normalizado | Regla_Condicional                                    |
|----------------|-------------|-------------|-------------|---------------|----------------------|------------------|------------------------------------------------------|
| 27             | 21-30       | 21          | 30          | hazard_global | hazard_core_global   | 1.42             | SI recencia in [21,30] ENTONCES subir peso (global). |

## Notas operativas
- Genera siempre parquet y csv (opcional) en `output_dir/`.
- No trunca: todos los activadores seleccionados se preservan.
- Si no hay patrones válidos, el script sigue corriendo y emite archivos vacíos con columnas esperadas (evita romper pipelines aguas abajo).
