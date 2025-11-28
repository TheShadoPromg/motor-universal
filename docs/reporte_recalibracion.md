# Informe de estado y recalibración del motor (estructural + hazard)

Fecha: 2025-11-27  
Autor: Codex (asistente)

## 1. Panorama actual (vigente)

- **Datos y preprocesos**
  - Fuente: `data/raw/eventos_numericos.*` (CSV/Parquet).
  - Validaciones: `great_expectations/` (vigente si se usa en pipelines; revisar uso real).
  - Orquestación/despliegue: `flows/`, `ops/`, `audit_pipeline-deployment.yaml`, `daily_pipeline-deployment.yaml` (requieren verificación de entorno/credenciales).

- **Auditoría estructural**
  - Fase 2 (`engine/audit/estructural.py`): transiciones (número, espejo, consecutivo, pos/any). Salidas en `data/audit/estructural/`.
  - Fase 2.5 (`engine/audit/estructural_fase2_5.py`): estabilidad por periodos (2011_2014, 2015_2018, 2019_2022, 2023_2025), clasifica core/extended/periodico. Salidas en `data/audit/estructural_fase2_5/`.
  - Fase 3 (`engine/audit/estructural_fase3_activadores.py`): activadores dinámicos core+periódico. Salidas: `data/activadores/activadores_dinamicos_fase3_para_motor.parquet|csv`.

- **Auditoría hazard/recencia (corregida)**
  - Fase 2.H (`engine/audit/hazard_recencia.py`): recencia por sorteo (ya no por fila), bins por defecto 1-5,6-10,11-20,21-30,31-45,46-60,61-90, estabilidad por subventanas con alpha configurable, signos consistentes requeridos también en extended/periodico. Salidas en `data/audit/hazard_train/`:
    - `hazard_global_resumen.*`, `hazard_numero_resumen.*`, `hazard_opportunities.*`.
  - Fase 3.H (`engine/audit/hazard_activadores.py`): pesos normalizados de forma conjunta (global + número), evita aplicar pesos de hazard a “todos los números”. Salidas en `data/activadores/hazard/activadores_hazard_para_motor.parquet|csv`.

- **Evaluador y tuning**
  - `engine/backtesting/phase4.py`: modelos A_uniforme, B_core, C_core_periodico, H_hazard, H_hazard_struct.
  - `engine/backtesting/phase4_tune.py`: grids beta/lambda (por defecto beta {0.5,1,1.5,2}, lambda {0.5,0.7,0.85,1}), splits Train/Valid/Test, sensibilidad, segmentos y “run card”. Salidas en `data/backtesting/`:
    - `phase4_grid_valid.*`, `best_phase4_params.*`, `phase4_results_final.*`, `phase4_results_segments.*`, `phase4_sensitivity_test.*`, `phase4_run_card.json`.
  - Pipeline OOS (`engine/backtesting/phase5_oos.py`): reentrena Fases 2/2.5/3 y 2.H/3.H en Train y evalúa en Valid/Test.

- **Estado de edge (último tuning)**
  - Core+periódico es el edge principal. Mejores hiperparámetros: beta=1.0, lambda=0.85.
  - Hazard solo: ruido (HR@10 Valid ≈ core, Test por debajo de core; LogLoss ≈ uniforme).
  - Hazard+struct: aporte marginal (beta_h=0.5, lambda_h≈0.70, beta_struct=1.0). HR@10 Test 0.2930 vs 0.2955 de core+periódico; LogLoss similar.

- **Motores “capas” (derived_dynamic, cross, struct_daily, fusión)**
  - `engine/derived_dynamic/transform.py`: genera `data/derived/derived_dynamic.parquet` (reglas espejo/complemento/seq/sum_mod, lags {1,2,3,7,14,30}, ventanas short/mid/long con pesos 0.5/0.3/0.2). Normaliza eventos, asegura grilla completa, valida oportunidades mínimas. Puede subir a S3 (`DERIVED_DYNAMIC_BUCKET`/`PREFIX`).
  - `engine/derived_dynamic/aggregate_daily.py`: consolida a `derived_daily.parquet`, calcula `score_derivado` y detalle top reglas activas; soporta `--all-dates` (batch completo `derived_daily_all.parquet`, actualiza latest). Upload S3 opcional (`DERIVED_DAILY_BUCKET`).
  - `engine/cross/aggregate_daily.py`: capa cross que cuenta oportunidades/activaciones origen→destino por lags (1,2,3,7,14,30) y posiciones (1/2/3); soporta `--all-dates` (`cross_daily_all.parquet`, actualiza latest). GE opcional y upload S3 (`CROSS_DAILY_BUCKET`). Usa eventos normalizados de `derived_dynamic.transform`.
  - `engine/structural/aggregate_daily.py`: activadores estructurales vigentes (core+periódico) y softmax (beta, lambda). Soporta `--all-dates` (`struct_daily_all.parquet`, actualiza latest). Grilla completa 00-99; GE opcional; upload S3 (`STRUCT_DAILY_BUCKET`).
  - `engine/fusion/fusionar_3capas.py`: fusiona C (cross) + E (estructural diario) + D (derivado) con pesos (default 0.4/0.3/0.3), threshold y temperatura softmax. En batch usa la intersección real de fechas cross/struct/derived y omite fechas sin datos; soporta `--date-range`. Salidas: `jugadas_fusionadas_3capas.parquet` (latest) y `jugadas_fusionadas_3capas_batch.parquet`. Upload S3 opcional (`FUSION_BUCKET`).
  - `engine/fusion/produce_predictions.py`: toma la fusión, aplica top-K/umbral y temperatura para generar jugadas/predicciones; permite snapshot o latest `jugadas_fusionadas_3capas.parquet`.

- **Flujos de información (capas)**
  - Eventos crudos → `derived_dynamic.transform` → `derived_dynamic.parquet` (+ snapshots).
  - `derived_dynamic.aggregate_daily` → `derived_daily.parquet`.
  - `cross.aggregate_daily` (sobre eventos normalizados) → `cross_daily.parquet`.
  - `struct_daily` (agregado diario de scores estructurales; verificar origen) → `struct_daily.parquet`.
  - `fusionar_3capas` combina `cross_daily` + `struct_daily` + `derived_daily` → `jugadas_fusionadas_3capas.parquet` → `produce_predictions` (jugadas finales).

## 2. Legado / obsoleto

- Lógica hazard previa que aplicaba pesos de un bin a todos los números y normalizaba global/número por separado.
- Recencia medida por fila/posición (bug antiguo).
- Cualquier dependencia no usada de `data-fabric/`, `scripts/`, `apps/`, `api/`, `ui/` debe revisarse; si no están en producción, tratarlas como legado.
- Great Expectations: mantener sólo si se ejecuta en pipelines actuales; caso contrario, archivar.

## 3. Criterios de selección/peso (propuestos)

- **Selección de patrones**
  - Mínimo oportunidades por bin/par (configurable).
  - Δ_rel mínima > 0 (ajustada al motor): global hazard ≥0.05–0.10; número hazard ≥0.30–0.50; estructural según Fase 2.5.
  - p_val 1-cola ≤ alpha (global, número) y estabilidad por subventanas con alpha_subwindow (p.ej. 0.05).
  - Signos consistentes en subventanas (sin deltas claramente negativas).
  - Opcional: control de múltiples comparaciones (FDR sencillo) para nuevos motores con muchos tests.

- **Pesado**
  - Magnitud: `mag_factor = log1p(max(Δ_rel, 0))`.
  - Estabilidad: `stab_factor = 0.5 + 0.5 * stability_score`.
  - Clase: core > extended > periodico (p.ej. 1.5 / 1.2 / 1.0).
  - Peso_Bruto = class_factor * mag_factor * stab_factor.
  - Normalización conjunta por motor (media(Peso_Bruto>0) = 1.0) para mantener escala relativa.

- **Validación**
  - Splits temporales fijos (Train/Valid/Test) y, si hay datos, k-fold temporal interno para estimar varianza.
  - Sensibilidad: probar ±1 paso en beta/lambda alrededor del óptimo; descartar hiperparámetros frágiles.
  - Métricas pivote: LogLoss y HR@10; requerir mejora mínima OOS (p.ej. HR@10 +2–3% o ΔLogLoss < -0.01) vs baseline core+periódico.
  - Segmentos en Test (mitades, DOW) para detectar drift.

## 4. Hiperparámetros óptimos actuales (from `phase4_run_card.json`)

- Core (B_core): beta=1.0, lambda=0.50.
- Core+periódico (C_core_periodico): beta=1.0, lambda=0.85. **Modelo preferido.**
- Hazard (H_hazard): beta=0.5, lambda=0.50 (ruido; mantener apagado o con lambda→1).
- Hazard+struct (H_hazard_struct): beta_hazard=0.5, lambda_hazard=0.70, beta_struct=1.0 (aporta marginalmente; opcional).

## 5. Recomendaciones ejecutivas

1) **Modelo operativo:** usar core+periódico con betas/lambdas óptimos como baseline.  
2) **Hazard:** mantener desactivado o con lambda alto/beta bajo hasta que aparezca edge robusto; incluirlo solo en experimentos controlados.  
3) **Pipeline estándar:** institucionalizar `phase5_oos.py` (reentrenar en Train, tune en Valid, medir en Test) como fuente única de métricas.  
4) **Nuevos motores:** aplicar los criterios de selección/peso y la validación descrita; no desplegar si no superan baseline con margen.  
5) **Drift/monitor:** revisar `phase4_results_segments.*` y repetir tuning trimestral o cuando se actualicen datos.  
6) **Limpieza:** catalogar/archivar componentes no usados (Great Expectations, data-fabric, apps/api/ui) para reducir ruido operativo.
7) **Capas derivada/cross/fusión:** mantener pesos y lags configurables; validar que `struct_daily.parquet` proviene del modelo estructural core+periódico (beta=1.0, lambda=0.85). Fusión batch 2011-10-19 a 2025-06-04: HR@10 0.6054 vs baseline estructural 0.3151 (x1.92), ΔLogLoss -0.0100; mantener pesos 0.4/0.3/0.3 mientras sostenga ese edge.

## 6. Ubicación de artefactos clave

- Activadores estructurales: `data/activadores/activadores_dinamicos_fase3_para_motor.parquet`.
- Activadores hazard: `data/activadores/hazard/activadores_hazard_para_motor.parquet`.
- Auditoría hazard: `data/audit/hazard_train/` (global, número, opportunities).
- Tuning/reportes: `data/backtesting/phase4_*.*`, `data/backtesting/phase4_run_card.json`.
- Documentación: `docs/auditoria/` (hazard, activadores, mapa), `docs/backtesting/` (Fase4/Fase5).
- Capas derivada/cross/fusión: `data/derived/derived_dynamic*.parquet`, `derived_daily.parquet`/`derived_daily_all.parquet`, `cross_daily.parquet`/`cross_daily_all.parquet`, `struct_daily.parquet`/`struct_daily_all.parquet`, `jugadas_fusionadas_3capas*.parquet` (incluye batch).
- Run card fusión batch: `data/derived/fusion_run_card_batch_2011-10-19_2025-06-04.json` (pesos 0.4/0.3/0.3, temp=1.0, métricas HR@K/LogLoss vs baseline estructural).

## 7. Próximos pasos sugeridos

- Congelar hiperparámetros en producción según `phase4_run_card.json` (core+periódico).  
- Ejecutar `phase5_oos.py` con los nuevos activadores hazard si se quiere evaluar su aporte en ventanas OOS.  
- Programar recalibración periódica (p.ej. mensual/trimestral) con grids acotados y run card archivada.  
- Si se integran nuevos motores, añadirles run card y criterios de selección/peso antes de cualquier despliegue.
- Revisar y documentar la fuente de `struct_daily.parquet`; asegurarse de que refleje el modelo estructural vigente (core+periódico) antes de la fusión C+E+D. Ajustar pesos de fusión si no hay lift sobre el baseline.
