import re, pandas as pd
PATH = "data/raw/draws.csv"

df = pd.read_csv(PATH, dtype=str)  # todo como string para inspección
for c in ["pos1","pos2","pos3"]:
    if c not in df.columns: 
        continue
    s = df[c].astype(str)
    # valores “limpios” si son 1–2 dígitos (permitiendo espacios alrededor)
    ok = s.str.match(r"^\s*\d{1,2}\s*$", na=False)
    bad = s[~ok].value_counts(dropna=False).head(25)
    print(f"\n=== {c} ===")
    print("TOTAL filas:", len(s), "   MAL:", (~ok).sum())
    print("Top valores problemáticos:")
    print(bad if len(bad) else "(sin problemas)")
