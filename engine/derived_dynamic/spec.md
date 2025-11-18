# Derivado Dinámico — Especificación

## Principios (alineados a protocolos)
- No truncamiento: evaluar siempre los 100 números (00–99) y todas las fechas disponibles.
- Consistencia temporal: lags explícitos; ventanas móviles 90/180/360; nada de mezclar in-sample con out-of-sample.
- Trazabilidad: cada fila del artefacto indica relación, parámetros (k), lag, oportunidades, activaciones y consistencia.

## Entradas
- `draws(fecha, pos1, pos2, pos3)` con fechas completas y números "NN" (00–99).

## Universo y notación
- Números: `U = {00, 01, ..., 99}` (representación string de dos dígitos).
- Fechas ordenadas: `F = [t1 < t2 < ... < tT]`.
- Posiciones observadas por fecha: `S(t) = {pos1(t), pos2(t), pos3(t)}` (conjunto).

## Relaciones (tipo_relacion, k)
1. `espejo`: `mirror(n)` = invertir dígitos (p.ej., 37↔73; 08↔80; 00↔00; 10↔01).
2. `sum_mod`: `n ⟷ (n ± k) mod 100`, con `k ∈ {1,2,5,10,50}` (ambos signos).
3. `seq`: adyacencia circular: `n ⟷ (n±1) mod 100` (opcional extender a ±2, ±5).
4. `complemento`: `n ⟷ (100 - n) mod 100` (00↔00; 50↔50).

**Normalización de n:** operar como enteros modulo 100, devolver siempre string `zfill(2)`.

## Definición de Activación (por fecha t, número n, relación R, parámetro k si aplica, lag L)
- Sea `Antecedentes_R(n,k)` el conjunto de números que “disparan” a `n` según R y k.
  - Ej.: para `seq, k=1`: `Antecedentes(n) = { (n-1) mod 100, (n+1) mod 100 }`.
  - Para `espejo`: `Antecedentes(n) = { mirror(n) }`.
  - Para `sum_mod, k`: `{ (n-k) mod 100, (n+k) mod 100 }`.
  - Para `complemento`: `{ (100-n) mod 100 }`.

- **Activación en t con lag L**:
  - Si **existe** fecha `t-L` en el histórico:
    - `activacion(t,n,R,k,L) = 1` si `S(t-L) ∩ Antecedentes_R(n,k) ≠ ∅`.
    - En otro caso, `0`.
  - Si **no existe** `t-L`: la evaluación no es posible.

## Oportunidades reales
Para cada `(t,n,R,k,L)`:
- `oportunidad(t,n,R,k,L) = 1` **solo si** existe `t-L` (es decir, la condición pudo evaluarse).
- `0` en caso contrario.

> Intuición: un día dado aporta a lo sumo **1** oportunidad por combinación `(n,R,k,L)` (evaluación binaria “posible/no posible” en ese día).

## Consistencia por ventanas móviles
Para cada `(t,n,R,k,L)` y ventana `w ∈ {90,180,360}`:
- Considera el historial `H_w(t) = {τ | τ < t y t-τ ≤ w}`.
- Define:  
  `activaciones_w = Σ_{τ∈H_w(t)} activacion(τ,n,R,k,L)`  
  `oportunidades_w = Σ_{τ∈H_w(t)} oportunidad(τ,n,R,k,L)`
- Tasa por ventana:  
  `p_w = activaciones_w / oportunidades_w` (si `oportunidades_w=0`, define `p_w = 0`).
- **Consistencia**: combinación ponderada usando `DERIVED_WINDOW_WEIGHTS` (por defecto short=0.5, mid=0.3, long=0.2).  
  Solo aportan las ventanas que cuenten con oportunidades históricas (los pesos se re-normalizan entre ellas).  
  `consistencia(t,n,R,k,L) = (Σ w_w * p_w) / (Σ w_w disponibles)`.
- **Filtro de oportunidades**: se calcula `oportunidades_historial` sobre la ventana “long” y se exige que sea ≥ `MIN_OPORTUNIDADES` (30 por defecto) para considerar la combinación; en caso contrario se marca como `datos_suficientes = False` y no se promedia en el score derivado diario.

## Salida (dataset `derived_dynamic`)
Columnas:
- `fecha` (date, día evaluado `t`)
- `numero` (string "NN")
- `tipo_relacion` ∈ {`espejo`,`sum_mod`,`seq`,`complemento`}
- `k` (int | null; null para `espejo` y `complemento`, 1 para `seq`, conjunto definido para `sum_mod`)
- `lag` (int ≥ 1)
- `oportunidades` (int ≥ 0) —= `oportunidad(t,...)`
- `activaciones` (int ≥ 0) —= `activacion(t,...)`
- `consistencia` (float 0..1) — calculada con ventanas
- `oportunidades_historial` (int ≥ 0) — oportunidades acumuladas en la ventana larga antes de filtrar.
- `datos_suficientes` (bool) — indica si la combinación pasa el mínimo y puede contribuir al score agregado.

## Reglas adicionales
- Siempre producir exactamente 100×|R|×|K_R|×|L| filas por fecha (no truncar 00–99).
- `activaciones ≤ oportunidades` por definición.
- Lags recomendados iniciales: `{1,2,3,7,14,30}`.
- Conservar idempotencia: mismo input → mismo output.

## Ejemplo (intuición)
- Si `t=2025-01-10`, `n=37`, `R=espejo`, `L=1`.
- `Antecedentes= {73}`. Si el 2025-01-09 salió 73 en cualquier posición, entonces `activaciones=1, oportunidades=1`. Si no, `0/1`. Si no existe 2025-01-09 en el histórico, `0/0` y la fila de t llevará `oportunidades=0, activaciones=0, consistencia=0`.
