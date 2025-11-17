# engine/_utils/schema.py
from __future__ import annotations
import re
import unicodedata
from typing import Dict, List
import pandas as pd
import numpy as np

# Canonical names we want internally
CANON = ["date", "number", "pos1", "pos2", "pos3"]

# Accepted aliases for header matching
ALIASES: Dict[str, List[str]] = {
    "date":   ["date", "fecha", "day", "dt"],
    "number": ["number", "numero", "num", "número", "nro", "n"],
    # Either we have pos1/pos2/pos3 ... OR we have a single "position" column
    "pos1":   ["pos1", "e_pos1", "p1", "hit_pos1", "is_pos1"],
    "pos2":   ["pos2", "e_pos2", "p2", "hit_pos2", "is_pos2"],
    "pos3":   ["pos3", "e_pos3", "p3", "hit_pos3", "is_pos3"],
}
# Optional alias (long format)
POSITION_ALIASES = ["position", "pos", "posicion", "posición"]

# Common textual labels mapping to numeric positions
POSITION_VALUE_MAP = {
    "primero": "1",
    "primer": "1",
    "1er": "1",
    "first": "1",
    "segundo": "2",
    "segun": "2",
    "2do": "2",
    "second": "2",
    "tercero": "3",
    "tercer": "3",
    "3ro": "3",
    "third": "3",
    "pos1": "1",
    "pos2": "2",
    "pos3": "3",
    "p1": "1",
    "p2": "2",
    "p3": "3",
}

def _normalize_header(name: str) -> str:
    return re.sub(r"\s+", "", name).strip().lower()


def _normalize_token(value: object) -> str:
    """Normalize textual tokens (strip accents/spaces) before mapping."""
    text = "" if value is None else str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", "", text)

def build_column_map(cols: list[str]) -> Dict[str, str]:
    """Return a mapping {canonical_name: original_name} for wide format.
       Raises if any of the required pos1/pos2/pos3 columns are missing."""
    norm = {c: _normalize_header(c) for c in cols}
    inv  = {v: k for k, v in norm.items()}
    mapping: Dict[str, str] = {}
    for canon, candidates in ALIASES.items():
        found = None
        for cand in candidates:
            key = _normalize_header(cand)
            if key in inv:
                found = inv[key]
                break
        if not found:
            raise KeyError(f"Missing required column for '{canon}'. Provided columns: {cols}")
        mapping[canon] = found
    return mapping

def detect_long_format(cols: list[str]) -> bool:
    """True if there's a single 'position' column instead of pos1/pos2/pos3."""
    norm = {_normalize_header(c) for c in cols}
    has_position = any(_normalize_header(p) in norm for p in POSITION_ALIASES)
    has_any_pos = any(_normalize_header(a) in norm for a in (ALIASES["pos1"]+ALIASES["pos2"]+ALIASES["pos3"]))
    return has_position and not has_any_pos

def normalize_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts either:
       - long:  columns ~ [date, number, position]
       - wide:  columns ~ [date, number, pos1, pos2, pos3]
       Returns canonical wide dataframe with columns: date, number, pos1, pos2, pos3.
    """
    cols = list(df.columns)
    norm = {_normalize_header(c): c for c in cols}

    # map date/number
    date_col = next((norm[_normalize_header(c)] for c in ["date","fecha","day","dt"] if _normalize_header(c) in norm), None)
    number_col = next((norm[_normalize_header(c)] for c in ["number","numero","num","número","nro","n"] if _normalize_header(c) in norm), None)
    if not date_col or not number_col:
        raise KeyError(f"CSV must contain date & number columns. Found: {cols}")

    # LONG format?
    if detect_long_format(cols):
        pos_col = next((norm[_normalize_header(c)] for c in POSITION_ALIASES if _normalize_header(c) in norm), None)
        if pos_col is None:
            raise KeyError("Could not find 'position' column.")
        out = df[[date_col, number_col, pos_col]].copy()
        out.columns = ["date", "number", "position"]
        # clean types
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["number"] = out["number"].astype(str).str.strip().str.replace(r"[^\d]", "", regex=True).replace("", np.nan)
        out["number"] = out["number"].astype(float).astype("Int64")  # allow NaN
        out["number"] = out["number"].fillna(0).astype(int).map(lambda x: f"{x:02d}")
        normalized_pos = out["position"].map(_normalize_token)
        normalized_pos = normalized_pos.replace(POSITION_VALUE_MAP)
        out["position"] = pd.to_numeric(normalized_pos, errors="coerce").fillna(0).astype(int)

        # explode to binary indicators per date,number
        out["pos1"] = (out["position"] == 1).astype(int)
        out["pos2"] = (out["position"] == 2).astype(int)
        out["pos3"] = (out["position"] == 3).astype(int)
        out = out.groupby(["date","number"], as_index=False)[["pos1","pos2","pos3"]].max()

        # ensure full grid (00-99) per date: fill missing with 0s
        all_dates = out["date"].drop_duplicates().sort_values()
        all_numbers = pd.Index([f"{i:02d}" for i in range(100)], name="number")
        frames = []
        for d in all_dates:
            base = pd.DataFrame({"date": d, "number": all_numbers})
            merged = base.merge(out[out["date"]==d], on=["date","number"], how="left")
            for c in ["pos1","pos2","pos3"]:
                merged[c] = merged[c].fillna(0).astype(int)
            frames.append(merged)
        wide = pd.concat(frames, ignore_index=True)
        return wide.sort_values(["date","number"]).reset_index(drop=True)

    # WIDE format (already has pos1/pos2/pos3)
    mapping = build_column_map(cols)  # will raise if missing
    wide = df.rename(columns={
        mapping["date"]: "date",
        mapping["number"]: "number",
        mapping["pos1"]: "pos1",
        mapping["pos2"]: "pos2",
        mapping["pos3"]: "pos3",
    }).copy()
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce")
    wide["number"] = wide["number"].astype(str).str.strip().str.replace(r"[^\d]", "", regex=True).replace("", np.nan)
    wide["number"] = wide["number"].astype(float).astype("Int64").fillna(0).astype(int).map(lambda x: f"{x:02d}")
    for c in ["pos1","pos2","pos3"]:
        wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0).astype(int).clip(0,1)
    return wide.sort_values(["date","number"]).reset_index(drop=True)
