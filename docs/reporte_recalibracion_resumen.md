# Resumen ejecutivo y visual (recalibración del motor)

Fecha: 2025-11-27  
Ámbito: estructural, hazard, derived_dynamic/cross/fusión, backtesting/tuning. Batch completo recalculado.

## 1) Qué está vigente (en 30s)
- Estructural: Fase 2/2.5/3 → activadores dinámicos en `data/activadores/activadores_dinamicos_fase3_para_motor.parquet`.
- Hazard (corregido): Fase 2.H/3.H → activadores en `data/activadores/hazard/activadores_hazard_para_motor.parquet` (recencia por sorteo, pesos normalizados conjunto).
- Backtesting/tuning: `phase4_tune` con modelos A/B/C/H/HS, grids beta/lambda, run card en `data/backtesting/phase4_run_card.json`.
- Capas diarias (batch listo): `derived_dynamic` -> `derived_daily.parquet` + `derived_daily_all.parquet`; `cross_daily.parquet` + `cross_daily_all.parquet`; `struct_daily.parquet` + `struct_daily_all.parquet` (activadores core+periódico vía softmax). Fusión 3 capas -> `jugadas_fusionadas_3capas.parquet` + `jugadas_fusionadas_3capas_batch.parquet`.

## 2) Flujos de información (vista rápida)
```
eventos_numericos
   ├─ auditoría estructural (F2/2.5/3) → activadores estructurales
   ├─ hazard (F2.H/3.H) → activadores hazard
   ├─ derived_dynamic.transform → derived_dynamic.parquet → aggregate_daily → derived_daily.parquet
   ├─ cross.aggregate_daily (mismo panel normalizado) → cross_daily.parquet
   ├─ struct_daily (activadores core+periódico → softmax) → struct_daily.parquet
   └─ fusión (cross + struct + derived) → jugadas_fusionadas_3capas.parquet → produce_predictions
```

## 3) Resultados recientes (Valid/Test) – HR@10 y LogLoss
```
HR@10 (Valid/Test)
Uniforme          0.2619 / 0.2344  |███████
Core              0.2667 / 0.2405  |████████
Core+Periódico    0.3097 / 0.2955  |███████████
Hazard            0.2659 / 0.2357  |████████
Hazard+Struct     0.3121 / 0.2930  |███████████

LogLoss (Valid/Test) – más bajo es mejor
Uniforme          4.6052 / 4.6052
Core              4.6038 / 4.6045
Core+Periódico    4.5911 / 4.5983
Hazard            4.6053 / 4.6052
Hazard+Struct     4.5913 / 4.5976
```
(Fuente: `data/backtesting/phase4_results_final.parquet`, hiperparámetros en `phase4_run_card.json`)

## 4) Edge vs. ruido
- Edge claro: Core+Periódico (C).  
- Edge marginal: Hazard+Struct (HS) - mejora ínfima; usar solo en pruebas.  
- Ruido: Hazard solo (H); no supera core.  
- Fusión 3 capas (batch 2011-10-19 a 2025-06-04): HR@10 0.6054 vs baseline estructural 0.3151 (x1.92), ΔLogLoss -0.0100. Pesos 0.4/0.3/0.3, temp=1.0 (ver `data/derived/fusion_run_card_batch_2011-10-19_2025-06-04.json`).

## 5) Recomendaciones accionables
1. Operar con Core+Periódico (beta=1.0, lambda=0.85).  
2. Mantener Hazard apagado o con lambda alto/beta bajo hasta ver edge robusto; HS solo en experimentos.  
3. Alinear `struct_daily.parquet` al modelo estructural vigente (C) antes de fusionar con cross/derived.  
4. Medir fusión (C+E+D) contra baseline C usando HR@K y LogLoss; batch actual supera el umbral (HR@10 x1.92, ΔLogLoss -0.0100). Mantener pesos 0.4/0.3/0.3 mientras sostenga el edge; recalibrar si cambia.  
5. Institucionalizar `phase5_oos.py` como pipeline estándar (Train→Valid→Test) y recalibrar trimestralmente con run card archivada.  
6. Limpiar/archivar componentes no usados (Great Expectations, data-fabric, apps/api/ui) si no están en producción.  
7. Registrar “run card” diaria para fusión (fecha, pesos C/E/D, temp, inputs `cross_daily/struct_daily/derived_daily`) para trazabilidad.

## 6) Dónde mirar
- Run card (hiperparámetros y métricas): `data/backtesting/phase4_run_card.json`.
- Grid/sensibilidad/segmentos: `data/backtesting/phase4_*.*`.
- Activadores vigentes: `data/activadores/activadores_dinamicos_fase3_para_motor.parquet`, `data/activadores/hazard/activadores_hazard_para_motor.parquet`.
- Capas diarias y fusión: `data/derived/derived_daily.parquet` + `_all`, `cross_daily.parquet` + `_all`, `struct_daily.parquet` + `_all`, `jugadas_fusionadas_3capas*.parquet` (incluye batch).
- Run card fusión batch: `data/derived/fusion_run_card_batch_2011-10-19_2025-06-04.json`.
