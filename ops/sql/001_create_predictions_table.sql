CREATE TABLE IF NOT EXISTS predictions_daily (
    fecha DATE NOT NULL,
    numero TEXT NOT NULL,
    score_cruzado DOUBLE PRECISION,
    score_estructural DOUBLE PRECISION,
    score_derivado DOUBLE PRECISION,
    score_total DOUBLE PRECISION,
    prob DOUBLE PRECISION,
    tipo_convergencia TEXT,
    detalles TEXT,
    prob_raw DOUBLE PRECISION,
    rank INTEGER,
    PRIMARY KEY (fecha, numero)
);
