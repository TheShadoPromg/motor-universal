-- Tabla para activar reglas estructurales dinámicas derivadas de Fase 3.
CREATE TABLE IF NOT EXISTS activadores_dinamicos_fase3 (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    ingestion_ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    numero_objetivo INTEGER NOT NULL,
    pos_origen TEXT,
    pos_destino TEXT,
    lag INTEGER NOT NULL,
    numero_condicionante INTEGER NOT NULL,
    tipo_relacion TEXT NOT NULL,
    clasificacion_fase2_5 TEXT NOT NULL,
    peso_bruto DOUBLE PRECISION,
    peso_normalizado DOUBLE PRECISION,
    regla_condicional TEXT,
    stability_score DOUBLE PRECISION,
    periodos_fuertes TEXT,
    UNIQUE (run_date, numero_objetivo, numero_condicionante, tipo_relacion, lag, pos_origen, pos_destino)
);

CREATE INDEX IF NOT EXISTS idx_activadores_objetivo ON activadores_dinamicos_fase3 (numero_objetivo);
CREATE INDEX IF NOT EXISTS idx_activadores_relacion ON activadores_dinamicos_fase3 (tipo_relacion, lag);
