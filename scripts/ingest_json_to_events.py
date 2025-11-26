# scripts/ingest_json_to_events.py
import json, re, glob, os
from datetime import datetime
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUT_CSV = os.path.join("data", "processed", "eventos_numericos.csv")

def split_premios(s):
    # ejemplo "64-36-30" o con espacios "73-70-48 "
    parts = [p.strip() for p in str(s).split("-")]
    while len(parts) < 3:
        parts.append("")  # por seguridad
    # normalizar a 2 dígitos 00-99
    norm = []
    for p in parts[:3]:
        p = re.sub(r"\D", "", p)  # quitar basura
        if p == "":
            norm.append("")
        else:
            norm.append(p.zfill(2)[-2:])
    return norm

def main():
    rows = []
    for path in glob.glob(os.path.join(RAW_DIR, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sorteos = data.get("sorteos", [])
        for it in sorteos:
            fecha = it.get("fecha_sorteo")  # "YYYY-MM-DD"
            premios = it.get("premios")
            pos1, pos2, pos3 = split_premios(premios)
            loteria_id = it.get("loteria_id")
            nombre = it.get("nombre_loteria")
            rows.append(
                {
                    "fecha": fecha,
                    "pos1": pos1,
                    "pos2": pos2,
                    "pos3": pos3,
                    "loteria_id": loteria_id,
                    "nombre_loteria": nombre,
                }
            )

    if not rows:
        raise SystemExit("No se encontraron sorteos en data/raw/*.json")

    df = pd.DataFrame(rows)

    # limpieza básica
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df = df.dropna(subset=["fecha"])
    # llave única (fecha, nombre_loteria)
    df = df.sort_values(["fecha", "nombre_loteria"]).drop_duplicates(
        subset=["fecha", "nombre_loteria"], keep="last"
    )

    # validar formato de pos1-3: dos dígitos o vacío
    for c in ["pos1", "pos2", "pos3"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
        df[c] = df[c].apply(lambda x: x.zfill(2)[-2:] if x.isdigit() else "")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Generado {OUT_CSV} con {len(df)} filas")

if __name__ == "__main__":
    main()
"""Ingesta JSON de sorteos a CSV canónico de eventos (date, position, number)."""
