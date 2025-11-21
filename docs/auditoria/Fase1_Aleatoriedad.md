# Auditoría de Aleatoriedad – Fase 1

**Lotería Real (RD) – Histórico completo (4,541 sorteos)**
**Versión:** 2025-06-05
**Estado:** ✅ Fase cerrada (baseline establecido)

---

## 0. Resumen ejecutivo

Objetivo de la Fase 1:

> Verificar si el histórico de la Lotería Real se comporta, en sus métricas básicas, de forma compatible con un modelo aleatorio uniforme (3 números distintos entre 00–99 por sorteo, con independencia día a día).

Conclusión:

* A nivel de:

  * frecuencias por número (00–99),
  * posición (1ro, 2do, 3ro),
  * par/impar,
  * alto/bajo (00–49 vs 50–99),
  * decenas,
  * repetición día a día,
  * reaparición condicional media por número,
  * rachas de paridad, rango y repetición,

el comportamiento observado es **altamente compatible** con el modelo aleatorio ideal.
No se detectan **sesgos gruesos** explotables con reglas simples de conteo.

Fase 1 queda cerrada y se declara como **baseline estadístico** del proyecto.

---

## 1. Datos y modelo nulo

### 1.1. Dataset analizado

* Archivo base: `eventos_numericos.csv`
* Estructura lógica:

  * `fecha`  → fecha del sorteo
  * `posición` → {1, 2, 3}
  * `número` → entero 00–99
* Volumen:

  * Sorteos: **4,541**
  * Eventos (número/posición): **13,623** (= 4,541 × 3)

### 1.2. Modelo nulo (baseline)

Se asume como referencia:

1. Cada sorteo es una selección **sin reemplazo** de 3 números distintos de {00,…,99}.
2. Todos los tríos posibles tienen la misma probabilidad.
3. Los sorteos de días distintos son **independientes** entre sí.

Sobre este modelo se calculan:

* esperados de frecuencia por número/posición,
* proporciones esperadas (par/impar, alto/bajo, decenas),
* probabilidades teóricas de repetición y reaparición.

---

## 2. Pruebas ejecutadas

### 2.1. Frecuencia global por número (00–99)

**Archivos de salida:**

* `frecuencia_global_resumen_2025-06-05.csv`

**Procedimiento:**

1. Contar apariciones de cada número 00–99 en todas las posiciones.
2. Valor esperado por número:
   [
   E = \frac{13{,}623}{100} \approx 136.23
   ]
3. Calcular estadístico χ²:
   [
   \chi^2 = \sum_{n=0}^{99} \frac{(O_n - E)^2}{E}
   ]
4. Comparar contra χ² con 99 grados de libertad.

**Resultados clave:**

* χ² ≈ **84.58**
* gl = 99
* p-value ≈ **0.8489**
* `desviacion_significativa_bool = False`

**Interpretación:**

* Las diferencias entre frecuencias observadas y esperadas son coherentes con ruido aleatorio.
* No hay evidencia de sesgos fuertes en la frecuencia global de ningún número.

---

### 2.2. Frecuencia por posición (1ro, 2do, 3ro)

**Archivos:**

* `frecuencia_por_posicion_resumen_2025-06-05.csv`

**Procedimiento:**

1. Repetir el análisis anterior condicionando por `posición`.
2. Para cada posición:

   * conteos por número,
   * χ² contra distribución uniforme en esa posición.

**Resultados (p-values aproximados):**

* Posición 1: p ≈ **0.648**
* Posición 2: p ≈ **0.457**
* Posición 3: p ≈ **0.945**
* En todos los casos: `desviacion_significativa_bool = False`.

**Interpretación:**

* Cada posición individual muestra una distribución de números compatible con uniformidad.
* No hay números “prohibidos” o “preferidos” por posición a escala global.

---

### 2.3. Par / Impar (global y por posición)

**Archivos:**

* `par_impar_global_resumen_2025-06-05.csv`
* `par_impar_por_posicion_resumen_2025-06-05.csv`

**Procedimiento:**

1. Reclasificar cada número como `par` o `impar`.
2. Comparar:

   * conteos globales par vs impar,
   * conteos par/impar por posición,
     con la proporción esperada 50/50 mediante χ².

**Resultados:**

* Global:

  * χ² ≈ **0.046**
  * p ≈ **0.83**
* Por posición:

  * p ≈ **0.96**, **0.71**, **0.96**
* Sin desviaciones significativas.

**Interpretación:**

* La paridad está extremadamente balanceada.
* No hay sesgo tipo “salen más pares/impares”.

---

### 2.4. Alto / Bajo (00–49 vs 50–99)

**Archivos:**

* `alto_bajo_global_resumen_2025-06-05.csv`
* `alto_bajo_por_posicion_resumen_2025-06-05.csv`

**Procedimiento:**

1. Clasificar:

   * `bajo` = 00–49
   * `alto` = 50–99
2. Comparar proporciones observadas vs 50/50 (global y por posición).

**Resultados (p-values):**

* Global: p ≈ **0.966**
* Posición 1: p ≈ **0.99**
* Posición 2: p ≈ **0.78**
* Posición 3: p ≈ **0.85**

**Interpretación:**

* No hay sesgos significativos en el balance alto/bajo.
* Estrategias basadas en “jugar más altos/bajos” no tienen soporte en el histórico total.

---

### 2.5. Decenas (00s, 10s, …, 90s)

**Archivos:**

* `decenas_global_resumen_2025-06-05.csv`
* `decenas_por_posicion_resumen_2025-06-05.csv`

**Procedimiento:**

1. Agrupar números por decena:

   * 00–09, 10–19, …, 90–99.
2. Aplicar χ² comparando distribución observada vs uniforme:

   * global,
   * por posición.

**Resultados:**

* Global:

  * χ² ≈ **7.65**, gl = 9
  * p ≈ **0.57**
* Por posición:

  * p ≈ **0.69**, **0.53**, **0.30**

**Interpretación:**

* Las decenas están distribuidas de forma compatible con aleatoriedad.
* No se identifica ninguna “decena sistemáticamente favorecida” en todo el período.

---

### 2.6. Repetición de números en días consecutivos

**Archivos:**

* `repeticion_dias_consecutivos_2025-06-05.csv`
* `repeticion_dias_consecutivos_resumen_2025-06-05.csv`

**Definiciones:**

* Para cada par de días consecutivos (D, D+1):

  * se cuenta cuántos números se repiten entre los 3 de D y los 3 de D+1 (0, 1, 2 o 3 repetidos).

**Datos observados:**

* Pares de días consecutivos: **4,540**
* Distribución:

  * 0 repetidos: 4,124
  * 1 repetido: 404
  * 2 repetidos: 11
  * 3 repetidos: 1
* Probabilidad empírica de **al menos 1 repetido**:
  [
  \hat{p}_\text{rep_emp} = \frac{404 + 11 + 1}{4,540} \approx 0.0916 \ (\text{9.16%})
  ]

**Modelo teórico (aleatorio ideal):**

* P(no repetidos en D+1) = (97/100)·(96/99)·(95/98) ≈ **0.9118**
* P(≥1 repetido) = 1 − 0.9118 ≈ **0.0882** (8.82%)

**Test χ² sobre 0–3 repetidos:**

* p-value ≈ **0.059**
* `desviacion_significativa_bool = False` al 5%.

**Interpretación:**

* La probabilidad empírica de repetición (≈9.16%) es muy cercana a la teórica (≈8.82%).
* El p ≈ 0.059 sugiere, como mucho, una **señal débil**; no suficiente para declarar un patrón sólido al 5% de significancia.
* No se justifica, con esta sola métrica, introducir una regla fuerte de “sobre-repetición diaria”.

---

### 2.7. Reaparición condicional por número

**Archivo:**

* `condicional_reaparicion_resumen_2025-06-05.csv`

**Definición:**

* Para cada número ( n ):

  * Se estima ( p_\text{emp_cond}(n) = P(n \text{ aparece en D+1} \mid n \text{ apareció en D}) ).

**Resumen estadístico:**

* Números: 100
* Media ( \bar{p}_\text{emp_cond} ) ≈ **0.031751**
* Desviación estándar ≈ **0.015729**
* p teórico (modelo ideal): **0.03**
* Diferencia media ≈ **0.00175**

**Interpretación:**

* La media empírica de reaparición condicional está prácticamente alineada con el 3% teórico.
* La dispersión entre números se explica por ruido.
* No se detecta ningún conjunto de números con reaparición claramente anómala en el histórico completo.

---

### 2.8. Rachas (par/impar, alto/bajo, repite/no repite)

**Archivos:**

* `rachas_par_impar_2025-06-05.csv`
* `rachas_par_impar_resumen_2025-06-05.csv`
* `rachas_alto_bajo_2025-06-05.csv`
* `rachas_alto_bajo_resumen_2025-06-05.csv`
* `rachas_repeticion_2025-06-05.csv`
* `rachas_repeticion_resumen_2025-06-05.csv`

**Procedimiento general:**

* Se toma una serie binaria en el tiempo:

  * par/impar,
  * alto/bajo,
  * repite/no_repite (respecto al día anterior).
* Se contabilizan:

  * número de rachas,
  * media de longitud de racha,
  * longitud máxima.

**Resultados clave:**

* **Par/Impar:**

  * Nº rachas ≈ 2,301
  * Longitud media ≈ **1.97**
  * Longitud máxima ≈ **12**
* **Alto/Bajo:**

  * Nº rachas ≈ 2,266
  * Longitud media ≈ **2.00**
  * Longitud máxima ≈ **13**
* **Repite/No repite:**

  * Nº rachas ≈ 752
  * Longitud media ≈ **6.04**
  * Longitud máxima ≈ **54** días con mismo estado.

**Interpretación:**

* Para procesos aproximadamente Bernoulli p≈0.5 (par/impar, alto/bajo), longitudes medias ~2 y máximos ~10–15 en varios miles de observaciones son normales. Los valores observados encajan bien.
* Para repite/no_repite, al ser “repite” un evento raro (~9%), es esperable observar rachas largas de “no_repite”. Las rachas observadas son coherentes con esa configuración.
* No se observan rachas que, por sí mismas, justifiquen la hipótesis de manipulación sistemática.

---

## 3. Conclusión global de la Fase 1

1. **Compatibilidad con aleatoriedad básica**

   * Todas las pruebas de primer orden (frecuencias, paridad, rangos, decenas, repetición global, reaparición media, rachas básicas) convergen en una misma conclusión:
   * El histórico 2011–2025 de la Lotería Real se comporta, en superficie, como un proceso **muy cercano** al modelo aleatorio uniforme e independiente.

2. **Ausencia de sesgos gruesos explotables**

   * No hay evidencia robusta de:

     * números “dopados” o “prohibidos”,
     * desbalances fuertes par/impar,
     * predominio estructural de altos/bajos,
     * decenas dominantes,
     * tasas de repetición diaria fuera de rango.

3. **Limitación explícita de la Fase 1**

   * La Fase 1 no analiza:

     * patrones condicionados finos por número y posición con lags específicos,
     * efectos por período (bloques de años),
     * estructuras derivadas (espejos, sumas modulares, secuencias) con dinámica temporal,
     * combinaciones de activadores.
   * Por diseño, Fase 1 actúa como **termómetro general**, no como detector fino de estructuras ocultas.

---

## 4. Implicaciones para fases siguientes del motor

1. **Descartadas como fuente principal de edge:**

   * Estrategias basadas únicamente en:

     * frecuencia global de números,
     * conteo simple par/impar,
     * alto/bajo,
     * decenas globales,
     * “repetición diaria” sin condicionamientos adicionales.

2. **Líneas de investigación justificadas para Fase 2+:**

   * Análisis condicional por número y posición con lags (matrices de transición).
   * Patrones derivados alineados a tus motores:

     * espejos,
     * sumas modulares (mod 10, mod 100),
     * secuencias,
     * combinaciones estructurales.
   * Segmentación temporal:

     * bloques de años,
     * cambios de comportamiento por períodos concretos,
     * posibles efectos por día de la semana.

3. **Uso de Fase 1 como baseline cuantitativo**

   * Cualquier regla/motor que se introduzca en Fase 2+ debe demostrar:

     * mejora clara frente a este baseline “casi aleatorio”,
     * estabilidad en subperíodos (evitar sobreajuste),
     * validación con métricas estrictas (logloss, Brier, lift frente a random, etc.).

---

## 5. Estado de la fase y artefactos generados

* **Estado:**

  * Fase 1 de Auditoría de Aleatoriedad: **COMPLETADA Y CERRADA**.
  * Se establece como referencia obligatoria para interpretar resultados futuros.

* **Archivos clave generados en la ejecución del 2025-06-05:**

  * `frecuencia_global_resumen_2025-06-05.csv`
  * `frecuencia_por_posicion_resumen_2025-06-05.csv`
  * `par_impar_global_resumen_2025-06-05.csv`
  * `par_impar_por_posicion_resumen_2025-06-05.csv`
  * `alto_bajo_global_resumen_2025-06-05.csv`
  * `alto_bajo_por_posicion_resumen_2025-06-05.csv`
  * `decenas_global_resumen_2025-06-05.csv`
  * `decenas_por_posicion_resumen_2025-06-05.csv`
  * `repeticion_dias_consecutivos_2025-06-05.csv`
  * `repeticion_dias_consecutivos_resumen_2025-06-05.csv`
  * `condicional_reaparicion_resumen_2025-06-05.csv`
  * `rachas_par_impar_2025-06-05.csv`
  * `rachas_par_impar_resumen_2025-06-05.csv`
  * `rachas_alto_bajo_2025-06-05.csv`
  * `rachas_alto_bajo_resumen_2025-06-05.csv`
  * `rachas_repeticion_2025-06-05.csv`
  * `rachas_repeticion_resumen_2025-06-05.csv`
