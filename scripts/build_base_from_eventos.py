"""Construye el dataset base eventos_numericos.parquet a partir del CSV raw."""

from pathlib import Path
import unicodedata

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "eventos_numericos.csv"
OUT_DIR = BASE_DIR / "data" / "base"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMN_ALIASES = {
    "fecha": ["fecha", "date"],
    "posicion": ["posicion", "posici\u00f3n", "posiciA3n", "posici\u00c3\u00b3n", "position"],
    "numero": [
        "numero",
        "n\u00famero",
        "nA\ufffdmero",
        "nA\ufffd\ufffdmero",
        "n\u00c3\u00bamero",
        "number",
    ],
}
EXPECTED_POSITIONS = {"primero", "segundo", "tercero"}
EXPECTED_FECHAS = 4541
NUMERO_MIN, NUMERO_MAX = 0, 99


def _normalize(text: str) -> str:
    """Normaliza un texto para facilitar comparaciones insensibles a acentos."""
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.strip().lower()


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra las columnas a los nombres canonicos esperados."""
    norm_to_actual = {_normalize(col): col for col in df.columns}
    rename_map = {}
    missing = []

    for canonical, aliases in EXPECTED_COLUMN_ALIASES.items():
        match = None
        for alias in aliases:
            alias_norm = _normalize(alias)
            if alias_norm in norm_to_actual:
                match = norm_to_actual[alias_norm]
                break
        if match is None:
            missing.append(canonical)
        else:
            rename_map[match] = canonical

    if missing:
        raise ValueError(f"No se encontraron las columnas requeridas: {', '.join(missing)}")

    return df.rename(columns=rename_map)


df = pd.read_csv(RAW_PATH)
df = canonicalize_columns(df)

df["fecha"] = pd.to_datetime(df["fecha"], errors="raise")
df["posicion"] = df["posicion"].astype(str).str.strip().str.lower()
df["numero"] = pd.to_numeric(df["numero"], errors="raise", downcast="integer")

# Validaciones fuertes (protocolo no truncamiento)
assert df["fecha"].nunique() == EXPECTED_FECHAS, f"Esperaba {EXPECTED_FECHAS:,} fechas"
position_counts = df["posicion"].value_counts()
assert set(position_counts.index) == EXPECTED_POSITIONS, "Posiciones inesperadas"
assert all(position_counts == EXPECTED_FECHAS), "Cada posicion debe tener 4,541 filas"
assert df["numero"].between(NUMERO_MIN, NUMERO_MAX).all(), "Numeros fuera del rango 00-99"

# Guardar version gobernada en parquet (formato largo)
df.to_parquet(OUT_DIR / "eventos_numericos.parquet", index=False)
print("OK. eventos_numericos.parquet generado como dataset base")
