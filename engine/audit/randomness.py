from __future__ import annotations

import argparse
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("audit.randomness")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "eventos_numericos.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "audit" / "randomness"
PARQUET_ENGINE = "pyarrow"
ALPHA = 0.05


@dataclass(frozen=True)
class AuditArtifacts:
    run_date: date
    freq_global: pd.DataFrame
    freq_global_summary: pd.DataFrame
    freq_pos: pd.DataFrame
    freq_pos_summary: pd.DataFrame
    par_impar_global: pd.DataFrame
    par_impar_global_summary: pd.DataFrame
    par_impar_pos: pd.DataFrame
    par_impar_pos_summary: pd.DataFrame
    alto_bajo_global: pd.DataFrame
    alto_bajo_global_summary: pd.DataFrame
    alto_bajo_pos: pd.DataFrame
    alto_bajo_pos_summary: pd.DataFrame
    decenas_global: pd.DataFrame
    decenas_global_summary: pd.DataFrame
    decenas_pos: pd.DataFrame
    decenas_pos_summary: pd.DataFrame
    repeticion_pairs: pd.DataFrame
    repeticion_summary: pd.DataFrame
    condicional: pd.DataFrame
    condicional_summary: pd.DataFrame
    rachas_par_impar: pd.DataFrame
    rachas_par_impar_summary: pd.DataFrame
    rachas_alto_bajo: pd.DataFrame
    rachas_alto_bajo_summary: pd.DataFrame
    rachas_repeticion: pd.DataFrame
    rachas_repeticion_summary: pd.DataFrame


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _chi2_sf(stat: float, df: int) -> Optional[float]:
    if df <= 0:
        return None
    if stat < 0:
        return None
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(stat, df))
    except Exception:
        LOGGER.warning("scipy.stats.chi2 no disponible; p-value se devuelve como None.")
        return None


def _ensure_int_number(raw: object) -> int:
    try:
        val = int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"No se pudo interpretar numero '{raw}'.") from exc
    if not (0 <= val <= 99):
        raise ValueError(f"Número fuera de rango 0-99: {val}")
    return val


def _normalize_position(raw: object) -> int:
    token = str(raw).strip().lower()
    mapping = {
        "1": 1,
        "01": 1,
        "primero": 1,
        "first": 1,
        "2": 2,
        "02": 2,
        "segundo": 2,
        "second": 2,
        "3": 3,
        "03": 3,
        "tercero": 3,
        "third": 3,
    }
    if token in mapping:
        return mapping[token]
    try:
        val = int(token)
    except Exception as exc:
        raise ValueError(f"No se pudo interpretar posicion '{raw}'.") from exc
    if val not in {1, 2, 3}:
        raise ValueError(f"Posicion fuera de rango 1-3: {val}")
    return val


def _read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de eventos: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def _normalize_events(raw_df: pd.DataFrame, start_date: Optional[date], end_date: Optional[date]) -> pd.DataFrame:
    df = raw_df.copy()
    col_map = {c.lower().strip(): c for c in df.columns}
    fecha_col = col_map.get("fecha") or col_map.get("date")
    numero_col = col_map.get("numero") or col_map.get("number")
    pos_col = col_map.get("posicion") or col_map.get("position")
    if not (fecha_col and numero_col and pos_col):
        raise ValueError("El dataset debe contener columnas fecha/date, numero/number y posicion/position.")

    df = df[[fecha_col, numero_col, pos_col]].rename(
        columns={fecha_col: "fecha", numero_col: "numero", pos_col: "posicion"}
    )
    df["numero"] = df["numero"].apply(_ensure_int_number)
    df["posicion"] = df["posicion"].apply(_normalize_position)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date

    if start_date:
        df = df[df["fecha"] >= start_date]
    if end_date:
        df = df[df["fecha"] <= end_date]

    if df["fecha"].isna().any():
        raise ValueError("Se encontraron fechas inválidas en eventos.")
    if df[["fecha", "posicion"]].duplicated().any():
        duplicated = df.loc[df[["fecha", "posicion"]].duplicated(), ["fecha", "posicion"]]
        raise ValueError(f"Duplicados detectados en (fecha, posicion):\n{duplicated.head()}")

    per_date_counts = df.groupby("fecha")["posicion"].nunique()
    bad_dates = per_date_counts[per_date_counts != 3]
    if not bad_dates.empty:
        raise ValueError(
            f"Cada fecha debe tener exactamente 3 posiciones. Fechas problemáticas: {bad_dates.index.tolist()}"
        )

    return df.sort_values(["fecha", "posicion"]).reset_index(drop=True)


def _add_run_date(df: pd.DataFrame, run_date: date) -> pd.DataFrame:
    df = df.copy()
    df["run_date"] = run_date.strftime("%Y-%m-%d")
    return df


def _test_frecuencia_global(df: pd.DataFrame, run_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = df["fecha"].unique()
    n_sorteos = len(dates)
    total_esperado = n_sorteos * 3 / 100 if n_sorteos > 0 else 0.0
    counts = df["numero"].value_counts().reindex(range(100), fill_value=0).sort_index()
    freq_rel = counts / (n_sorteos * 3) if n_sorteos > 0 else counts * 0.0

    residual = counts - total_esperado
    std_res = np.divide(residual, math.sqrt(total_esperado) if total_esperado > 0 else np.nan)

    table = pd.DataFrame(
        {
            "numero": [f"{i:02d}" for i in range(100)],
            "count_total": counts.to_numpy(),
            "esperado_total": total_esperado,
            "freq_relativa": freq_rel.to_numpy(),
            "desviacion_abs": residual.to_numpy(),
            "desviacion_rel": np.divide(residual, total_esperado) if total_esperado > 0 else np.zeros_like(residual),
            "residual_estandarizado": std_res,
        }
    )

    chi2 = float(((residual**2) / total_esperado).sum()) if total_esperado > 0 else 0.0
    p_value = _chi2_sf(chi2, df=99)
    summary = pd.DataFrame(
        [
            {
                "N_sorteos": n_sorteos,
                "total_observado": int(counts.sum()),
                "total_esperado": float(n_sorteos * 3),
                "chi2_global": chi2,
                "gl": 99,
                "p_value_global": p_value,
                "desviacion_significativa_bool": bool(p_value is not None and p_value < ALPHA),
            }
        ]
    )

    return _add_run_date(table, run_date), _add_run_date(summary, run_date)


def _test_frecuencia_posicion(df: pd.DataFrame, run_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = df["fecha"].unique()
    n_sorteos = len(dates)
    rows = []
    summary_rows = []
    for pos in (1, 2, 3):
        subset = df[df["posicion"] == pos]
        esperado = n_sorteos / 100 if n_sorteos > 0 else 0.0
        counts = subset["numero"].value_counts().reindex(range(100), fill_value=0).sort_index()
        freq_rel = counts / n_sorteos if n_sorteos > 0 else counts * 0.0
        residual = counts - esperado
        std_res = np.divide(residual, math.sqrt(esperado) if esperado > 0 else np.nan)
        for numero, cnt, fr, res, std in zip(counts.index, counts, freq_rel, residual, std_res):
            rows.append(
                {
                    "numero": f"{numero:02d}",
                    "posicion": pos,
                    "count_pos": int(cnt),
                    "esperado_pos": esperado,
                    "freq_relativa_pos": fr,
                    "desviacion_abs": res,
                    "desviacion_rel": res / esperado if esperado > 0 else 0.0,
                    "residual_estandarizado": std,
                }
            )

        chi2 = float(((residual**2) / esperado).sum()) if esperado > 0 else 0.0
        p_value = _chi2_sf(chi2, df=99)
        summary_rows.append(
            {
                "posicion": pos,
                "N_sorteos_pos": n_sorteos,
                "chi2_pos": chi2,
                "gl_pos": 99,
                "p_value_pos": p_value,
                "desviacion_significativa_bool": bool(p_value is not None and p_value < ALPHA),
            }
        )

    table = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    return _add_run_date(table, run_date), _add_run_date(summary, run_date)


def _category_counts(
    df: pd.DataFrame,
    label_col: str,
    label_values: Sequence[str],
    total: int,
    expected_each: float,
    run_date: date,
    extra_cols: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    counts = df[label_col].value_counts().reindex(label_values, fill_value=0)
    rows = []
    for label in label_values:
        cnt = counts[label]
        rows.append(
            {
                label_col: label,
                "count": int(cnt),
                "esperado": expected_each,
                "freq_relativa": cnt / total if total > 0 else 0.0,
                "desviacion_abs": cnt - expected_each,
                "desviacion_rel": (cnt - expected_each) / expected_each if expected_each > 0 else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    if extra_cols:
        for k, v in extra_cols.items():
            table[k] = v
    return _add_run_date(table, run_date)


def _chi2_from_table(table: pd.DataFrame, count_col: str = "count", expected_col: str = "esperado") -> float:
    observed = table[count_col].astype(float)
    expected = table[expected_col].astype(float)
    mask = expected > 0
    if not mask.any():
        return 0.0
    chi2 = float((((observed - expected) ** 2) / expected).loc[mask].sum())
    return chi2


def _categorias_global(
    df: pd.DataFrame, run_date: date
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(df)

    df = df.copy()
    df["categoria"] = np.where(df["numero"] % 2 == 0, "par", "impar")
    par_impar_table = _category_counts(
        df, "categoria", ["par", "impar"], total, total / 2 if total > 0 else 0.0, run_date
    )
    par_impar_table["categoria_tipo"] = "par_impar"
    par_impar_table = par_impar_table.rename(columns={"categoria": "categoria_valor"})
    par_impar_table = par_impar_table[
        ["categoria_tipo", "categoria_valor", "count", "esperado", "freq_relativa", "desviacion_abs", "desviacion_rel", "run_date"]
    ]
    chi2_par = _chi2_from_table(par_impar_table)
    p_value_par = _chi2_sf(chi2_par, df=1)
    par_impar_summary = _add_run_date(
        pd.DataFrame(
            [
                {
                    "categoria_tipo": "par_impar",
                    "total": total,
                    "chi2_par_impar": chi2_par,
                    "gl": 1,
                    "p_value": p_value_par,
                    "desviacion_significativa_bool": bool(p_value_par is not None and p_value_par < ALPHA),
                }
            ]
        ),
        run_date,
    )

    df["categoria"] = np.where(df["numero"] >= 50, "alto", "bajo")
    alto_bajo_table = _category_counts(
        df, "categoria", ["bajo", "alto"], total, total / 2 if total > 0 else 0.0, run_date
    )
    alto_bajo_table["categoria_tipo"] = "alto_bajo"
    alto_bajo_table = alto_bajo_table.rename(columns={"categoria": "categoria_valor"})
    alto_bajo_table = alto_bajo_table[
        ["categoria_tipo", "categoria_valor", "count", "esperado", "freq_relativa", "desviacion_abs", "desviacion_rel", "run_date"]
    ]
    chi2_alto = _chi2_from_table(alto_bajo_table)
    p_value_alto = _chi2_sf(chi2_alto, df=1)
    alto_bajo_summary = _add_run_date(
        pd.DataFrame(
            [
                {
                    "categoria_tipo": "alto_bajo",
                    "total": total,
                    "chi2_alto_bajo": chi2_alto,
                    "gl": 1,
                    "p_value": p_value_alto,
                    "desviacion_significativa_bool": bool(p_value_alto is not None and p_value_alto < ALPHA),
                }
            ]
        ),
        run_date,
    )

    df["decena"] = (df["numero"] // 10) * 10
    decena_values = list(range(0, 100, 10))
    decenas_table = _category_counts(df, "decena", decena_values, total, total / 10 if total > 0 else 0.0, run_date)
    decenas_table["decena"] = decenas_table["decena"].astype(int)
    decenas_table["scope"] = "global"
    decenas_table["posicion"] = pd.NA
    decenas_table = decenas_table[
        ["decena", "scope", "posicion", "count", "esperado", "freq_relativa", "desviacion_abs", "desviacion_rel", "run_date"]
    ]
    chi2_decenas = _chi2_from_table(decenas_table, count_col="count", expected_col="esperado")
    p_value_decenas = _chi2_sf(chi2_decenas, df=9)
    decenas_summary = _add_run_date(
        pd.DataFrame(
            [
                {
                    "total": total,
                    "chi2_decenas": chi2_decenas,
                    "gl": 9,
                    "p_value": p_value_decenas,
                    "desviacion_significativa_bool": bool(p_value_decenas is not None and p_value_decenas < ALPHA),
                }
            ]
        ),
        run_date,
    )

    return (
        par_impar_table,
        par_impar_summary,
        alto_bajo_table,
        alto_bajo_summary,
        decenas_table,
        decenas_summary,
    )


def _categorias_por_posicion(df: pd.DataFrame, run_date: date) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_por_pos = df["fecha"].nunique()
    par_rows = []
    par_summary_rows = []
    alto_rows = []
    alto_summary_rows = []
    decena_rows = []
    decena_summary_rows = []

    for pos in (1, 2, 3):
        subset = df[df["posicion"] == pos].copy()
        subset["categoria"] = np.where(subset["numero"] % 2 == 0, "par", "impar")
        par_table = _category_counts(
            subset,
            "categoria",
            ["par", "impar"],
            total_por_pos,
            total_por_pos / 2 if total_por_pos > 0 else 0.0,
            run_date,
            extra_cols={"posicion": pos},
        )
        par_table["categoria_tipo"] = "par_impar"
        par_table = par_table.rename(columns={"categoria": "categoria_valor"})
        par_table = par_table[
            ["categoria_tipo", "categoria_valor", "posicion", "count", "esperado", "freq_relativa", "desviacion_abs", "desviacion_rel", "run_date"]
        ]
        chi2_par = _chi2_from_table(par_table, count_col="count", expected_col="esperado")
        p_val_par = _chi2_sf(chi2_par, df=1)
        par_rows.append(par_table)
        par_summary_rows.append(
            {
                "posicion": pos,
                "categoria_tipo": "par_impar",
                "total": total_por_pos,
                "chi2_par_impar": chi2_par,
                "gl": 1,
                "p_value": p_val_par,
                "desviacion_significativa_bool": bool(p_val_par is not None and p_val_par < ALPHA),
                "run_date": run_date.strftime("%Y-%m-%d"),
            }
        )

        subset["categoria"] = np.where(subset["numero"] >= 50, "alto", "bajo")
        alto_table = _category_counts(
            subset,
            "categoria",
            ["bajo", "alto"],
            total_por_pos,
            total_por_pos / 2 if total_por_pos > 0 else 0.0,
            run_date,
            extra_cols={"posicion": pos},
        )
        alto_table["categoria_tipo"] = "alto_bajo"
        alto_table = alto_table.rename(columns={"categoria": "categoria_valor"})
        alto_table = alto_table[
            ["categoria_tipo", "categoria_valor", "posicion", "count", "esperado", "freq_relativa", "desviacion_abs", "desviacion_rel", "run_date"]
        ]
        chi2_alto = _chi2_from_table(alto_table, count_col="count", expected_col="esperado")
        p_val_alto = _chi2_sf(chi2_alto, df=1)
        alto_rows.append(alto_table)
        alto_summary_rows.append(
            {
                "posicion": pos,
                "categoria_tipo": "alto_bajo",
                "total": total_por_pos,
                "chi2_alto_bajo": chi2_alto,
                "gl": 1,
                "p_value": p_val_alto,
                "desviacion_significativa_bool": bool(p_val_alto is not None and p_val_alto < ALPHA),
                "run_date": run_date.strftime("%Y-%m-%d"),
            }
        )

        subset["decena"] = (subset["numero"] // 10) * 10
        decena_values = list(range(0, 100, 10))
        decena_table = _category_counts(
            subset,
            "decena",
            decena_values,
            total_por_pos,
            total_por_pos / 10 if total_por_pos > 0 else 0.0,
            run_date,
            extra_cols={"posicion": pos},
        )
        decena_table["decena"] = decena_table["decena"].astype(int)
        decena_table["scope"] = "posicion"
        decena_table = decena_table[
            ["decena", "scope", "posicion", "count", "esperado", "freq_relativa", "desviacion_abs", "desviacion_rel", "run_date"]
        ]
        chi2_dec = _chi2_from_table(decena_table, count_col="count", expected_col="esperado")
        p_val_dec = _chi2_sf(chi2_dec, df=9)
        decena_rows.append(decena_table)
        decena_summary_rows.append(
            {
                "posicion": pos,
                "total": total_por_pos,
                "chi2_decenas": chi2_dec,
                "gl": 9,
                "p_value": p_val_dec,
                "desviacion_significativa_bool": bool(p_val_dec is not None and p_val_dec < ALPHA),
                "run_date": run_date.strftime("%Y-%m-%d"),
            }
        )

    return (
        pd.concat(par_rows, ignore_index=True),
        pd.DataFrame(par_summary_rows),
        pd.concat(alto_rows, ignore_index=True),
        pd.DataFrame(alto_summary_rows),
        pd.concat(decena_rows, ignore_index=True),
        pd.DataFrame(decena_summary_rows),
    )


def _binom_pmf(k: int, n: int, p: float) -> float:
    if not (0 <= k <= n):
        return 0.0
    comb = math.comb(n, k)
    return comb * (p**k) * ((1 - p) ** (n - k))


def _repeticion_dias_consecutivos(date_sets: List[Tuple[date, Set[int]]], run_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    dist_counter = Counter()
    for idx in range(len(date_sets) - 1):
        fecha_t, nums_t = date_sets[idx]
        fecha_t1, nums_t1 = date_sets[idx + 1]
        repetidos = len(nums_t & nums_t1)
        dist_counter[repetidos] += 1
        rows.append({"fecha_t": fecha_t, "fecha_t1": fecha_t1, "repetidos_t_t1": repetidos})

    n_pairs = len(rows)
    total_repetidos = sum(dist_counter[k] * k for k in dist_counter)
    esperado_total = n_pairs * 0.09
    p_repeat = 0.03
    expected_counts = {k: n_pairs * _binom_pmf(k, 3, p_repeat) for k in range(4)}
    chi2 = 0.0
    for k in range(4):
        obs = dist_counter.get(k, 0)
        exp = expected_counts.get(k, 0.0)
        if exp > 0:
            chi2 += ((obs - exp) ** 2) / exp
    p_value = _chi2_sf(chi2, df=3)

    table = _add_run_date(pd.DataFrame(rows), run_date)
    summary = pd.DataFrame(
        [
            {
                "N_pairs": n_pairs,
                "total_repetidos": total_repetidos,
                "esperado_total_repetidos": esperado_total,
                "desviacion_abs": total_repetidos - esperado_total,
                "desviacion_rel": (total_repetidos - esperado_total) / esperado_total if esperado_total > 0 else 0.0,
                "count_repetidos_0": dist_counter.get(0, 0),
                "count_repetidos_1": dist_counter.get(1, 0),
                "count_repetidos_2": dist_counter.get(2, 0),
                "count_repetidos_3": dist_counter.get(3, 0),
                "expected_repetidos_0": expected_counts.get(0, 0.0),
                "expected_repetidos_1": expected_counts.get(1, 0.0),
                "expected_repetidos_2": expected_counts.get(2, 0.0),
                "expected_repetidos_3": expected_counts.get(3, 0.0),
                "chi2_repetidos": chi2,
                "gl": 3,
                "p_value": p_value,
                "desviacion_significativa_bool": bool(p_value is not None and p_value < ALPHA),
                "run_date": run_date.strftime("%Y-%m-%d"),
            }
        ]
    )
    return table, summary


def _condicional_reaparicion(date_sets: List[Tuple[date, Set[int]]], run_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    A = defaultdict(int)
    B = defaultdict(int)
    for idx in range(len(date_sets) - 1):
        nums_t = date_sets[idx][1]
        nums_t1 = date_sets[idx + 1][1]
        for num in nums_t:
            A[num] += 1
            if num in nums_t1:
                B[num] += 1

    rows = []
    p_teorico = 0.03
    for num in range(100):
        a = A.get(num, 0)
        if a == 0:
            continue
        b = B.get(num, 0)
        p_emp = b / a if a > 0 else 0.0
        rows.append(
            {
                "numero": f"{num:02d}",
                "A_n": a,
                "B_n": b,
                "p_emp_cond": p_emp,
                "p_teorico": p_teorico,
                "desviacion_abs": p_emp - p_teorico,
                "desviacion_rel": (p_emp - p_teorico) / p_teorico if p_teorico > 0 else 0.0,
            }
        )
    table = _add_run_date(pd.DataFrame(rows), run_date)
    if len(rows) > 0:
        mean_emp = float(table["p_emp_cond"].mean())
        std_emp = float(table["p_emp_cond"].std(ddof=0))
    else:
        mean_emp = 0.0
        std_emp = 0.0
    summary = _add_run_date(
        pd.DataFrame(
            [
                {
                    "n_numeros": len(rows),
                    "media_p_emp_cond": mean_emp,
                    "std_p_emp_cond": std_emp,
                    "p_teorico": p_teorico,
                    "diferencia_media": mean_emp - p_teorico,
                }
            ]
        ),
        run_date,
    )
    return table, summary


def _compute_runs(labels: List[str], dates: List[date]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not labels:
        return pd.DataFrame(columns=["fecha", "label", "run_id", "run_length"]), {"numero_rachas": 0, "media_largo_racha": 0.0, "max_largo_racha": 0}
    run_id = 1
    run_lengths: List[int] = []
    rows = []
    current_label = labels[0]
    current_length = 1
    for idx in range(1, len(labels)):
        if labels[idx] == current_label:
            current_length += 1
        else:
            run_lengths.append(current_length)
            current_label = labels[idx]
            current_length = 1
            run_id += 1
        rows.append({"fecha": dates[idx], "label": labels[idx], "run_id": run_id})
    run_lengths.append(current_length)
    rows.insert(0, {"fecha": dates[0], "label": labels[0], "run_id": 1})

    summary = {
        "numero_rachas": len(run_lengths),
        "media_largo_racha": float(np.mean(run_lengths)),
        "max_largo_racha": int(np.max(run_lengths)),
    }
    run_df = pd.DataFrame(rows)
    return run_df, summary


def _runs_par_impar(df: pd.DataFrame, run_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby("fecha")["numero"].apply(list).reset_index()
    labels = []
    for _, row in grouped.iterrows():
        nums = row["numero"]
        par = sum(1 for n in nums if n % 2 == 0)
        impar = len(nums) - par
        if par > impar:
            label = "mas_par"
        elif impar > par:
            label = "mas_impar"
        else:
            label = "mixto"
        labels.append(label)
    run_df, summary_stats = _compute_runs(labels, grouped["fecha"].tolist())
    run_df["scope"] = "par_impar"
    run_df = _add_run_date(run_df, run_date)
    summary = _add_run_date(
        pd.DataFrame([{**summary_stats}]),
        run_date,
    )
    return run_df, summary


def _runs_alto_bajo(df: pd.DataFrame, run_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby("fecha")["numero"].apply(list).reset_index()
    labels = []
    for _, row in grouped.iterrows():
        nums = row["numero"]
        alto = sum(1 for n in nums if n >= 50)
        bajo = len(nums) - alto
        if alto > bajo:
            label = "mas_alto"
        elif bajo > alto:
            label = "mas_bajo"
        else:
            label = "mixto"
        labels.append(label)
    run_df, summary_stats = _compute_runs(labels, grouped["fecha"].tolist())
    run_df["scope"] = "alto_bajo"
    run_df = _add_run_date(run_df, run_date)
    summary = _add_run_date(pd.DataFrame([summary_stats]), run_date)
    return run_df, summary


def _runs_repeticion(date_sets: List[Tuple[date, Set[int]]], run_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = []
    dates = [pair[0] for pair in date_sets]
    labels.append("sin_prev")  # primer día no tiene referencia
    for idx in range(1, len(date_sets)):
        prev_nums = date_sets[idx - 1][1]
        curr_nums = date_sets[idx][1]
        inter = len(prev_nums & curr_nums)
        label = "repite" if inter > 0 else "no_repite"
        labels.append(label)
    run_df, summary_stats = _compute_runs(labels, dates)
    run_df["scope"] = "repeticion"
    run_df = _add_run_date(run_df, run_date)
    summary = _add_run_date(pd.DataFrame([summary_stats]), run_date)
    return run_df, summary


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine=PARQUET_ENGINE)


def run_randomness_audit(
    run_date: date,
    input_path: str,
    output_dir: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    skip_validation: bool = False,
) -> AuditArtifacts:
    LOGGER.info("Iniciando auditoría de aleatoriedad (Fase 1) para run_date=%s", run_date)
    raw_df = _read_events(Path(input_path))
    events = _normalize_events(raw_df, start_date, end_date)
    if len(events) == 0:
        raise ValueError("El dataset de eventos está vacío después de filtrar por fechas.")
    dates_sorted = sorted(events["fecha"].unique())
    date_sets = [(dt, set(events.loc[events["fecha"] == dt, "numero"])) for dt in dates_sorted]
    LOGGER.info("Rango analizado: %s -> %s (%s sorteos)", dates_sorted[0], dates_sorted[-1], len(dates_sorted))

    freq_global, freq_global_summary = _test_frecuencia_global(events, run_date)
    freq_pos, freq_pos_summary = _test_frecuencia_posicion(events, run_date)

    (
        par_impar_global,
        par_impar_global_summary,
        alto_bajo_global,
        alto_bajo_global_summary,
        decenas_global,
        decenas_global_summary,
    ) = _categorias_global(events, run_date)

    (
        par_impar_pos,
        par_impar_pos_summary,
        alto_bajo_pos,
        alto_bajo_pos_summary,
        decenas_pos,
        decenas_pos_summary,
    ) = _categorias_por_posicion(events, run_date)

    repeticion_pairs, repeticion_summary = _repeticion_dias_consecutivos(date_sets, run_date)
    condicional_table, condicional_summary = _condicional_reaparicion(date_sets, run_date)

    rachas_par_impar, rachas_par_impar_summary = _runs_par_impar(events, run_date)
    rachas_alto_bajo, rachas_alto_bajo_summary = _runs_alto_bajo(events, run_date)
    rachas_repeticion, rachas_repeticion_summary = _runs_repeticion(date_sets, run_date)

    output_base = Path(output_dir)
    run_date_str = run_date.strftime("%Y-%m-%d")
    _save(freq_global, output_base / f"frecuencia_global_numeros_{run_date_str}.parquet")
    _save(freq_global_summary, output_base / f"frecuencia_global_resumen_{run_date_str}.parquet")
    _save(freq_pos, output_base / f"frecuencia_por_posicion_{run_date_str}.parquet")
    _save(freq_pos_summary, output_base / f"frecuencia_por_posicion_resumen_{run_date_str}.parquet")

    _save(par_impar_global, output_base / f"par_impar_global_{run_date_str}.parquet")
    _save(par_impar_global_summary, output_base / f"par_impar_global_resumen_{run_date_str}.parquet")
    _save(par_impar_pos, output_base / f"par_impar_por_posicion_{run_date_str}.parquet")
    _save(par_impar_pos_summary, output_base / f"par_impar_por_posicion_resumen_{run_date_str}.parquet")

    _save(alto_bajo_global, output_base / f"alto_bajo_global_{run_date_str}.parquet")
    _save(alto_bajo_global_summary, output_base / f"alto_bajo_global_resumen_{run_date_str}.parquet")
    _save(alto_bajo_pos, output_base / f"alto_bajo_por_posicion_{run_date_str}.parquet")
    _save(alto_bajo_pos_summary, output_base / f"alto_bajo_por_posicion_resumen_{run_date_str}.parquet")

    _save(decenas_global, output_base / f"decenas_global_{run_date_str}.parquet")
    _save(decenas_global_summary, output_base / f"decenas_global_resumen_{run_date_str}.parquet")
    _save(decenas_pos, output_base / f"decenas_por_posicion_{run_date_str}.parquet")
    _save(decenas_pos_summary, output_base / f"decenas_por_posicion_resumen_{run_date_str}.parquet")

    _save(repeticion_pairs, output_base / f"repeticion_dias_consecutivos_{run_date_str}.parquet")
    _save(repeticion_summary, output_base / f"repeticion_dias_consecutivos_resumen_{run_date_str}.parquet")
    _save(condicional_table, output_base / f"condicional_reaparicion_{run_date_str}.parquet")
    _save(condicional_summary, output_base / f"condicional_reaparicion_resumen_{run_date_str}.parquet")

    _save(rachas_par_impar, output_base / f"rachas_par_impar_{run_date_str}.parquet")
    _save(rachas_par_impar_summary, output_base / f"rachas_par_impar_resumen_{run_date_str}.parquet")
    _save(rachas_alto_bajo, output_base / f"rachas_alto_bajo_{run_date_str}.parquet")
    _save(rachas_alto_bajo_summary, output_base / f"rachas_alto_bajo_resumen_{run_date_str}.parquet")
    _save(rachas_repeticion, output_base / f"rachas_repeticion_{run_date_str}.parquet")
    _save(rachas_repeticion_summary, output_base / f"rachas_repeticion_resumen_{run_date_str}.parquet")

    LOGGER.info("Auditoría completada. Artefactos guardados en %s", output_base)

    return AuditArtifacts(
        run_date=run_date,
        freq_global=freq_global,
        freq_global_summary=freq_global_summary,
        freq_pos=freq_pos,
        freq_pos_summary=freq_pos_summary,
        par_impar_global=par_impar_global,
        par_impar_global_summary=par_impar_global_summary,
        par_impar_pos=par_impar_pos,
        par_impar_pos_summary=par_impar_pos_summary,
        alto_bajo_global=alto_bajo_global,
        alto_bajo_global_summary=alto_bajo_global_summary,
        alto_bajo_pos=alto_bajo_pos,
        alto_bajo_pos_summary=alto_bajo_pos_summary,
        decenas_global=decenas_global,
        decenas_global_summary=decenas_global_summary,
        decenas_pos=decenas_pos,
        decenas_pos_summary=decenas_pos_summary,
        repeticion_pairs=repeticion_pairs,
        repeticion_summary=repeticion_summary,
        condicional=condicional_table,
        condicional_summary=condicional_summary,
        rachas_par_impar=rachas_par_impar,
        rachas_par_impar_summary=rachas_par_impar_summary,
        rachas_alto_bajo=rachas_alto_bajo,
        rachas_alto_bajo_summary=rachas_alto_bajo_summary,
        rachas_repeticion=rachas_repeticion,
        rachas_repeticion_summary=rachas_repeticion_summary,
    )


def _parse_date(value: str, name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{name} debe tener formato YYYY-MM-DD") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoría Fase 1 de aleatoriedad sobre eventos_numericos.")
    parser.add_argument("--run-date", required=False, help="Fecha lógica de ejecución (YYYY-MM-DD).")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Ruta al CSV/Parquet de eventos_numericos.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio base para artefactos.")
    parser.add_argument("--start-date", default=None, help="Fecha inicial a incluir (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Fecha final a incluir (YYYY-MM-DD).")
    parser.add_argument("--skip-validation", action="store_true", help="Omite validaciones externas adicionales.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)
    run_date = _parse_date(args.run_date, "run-date") if args.run_date else datetime.utcnow().date()
    start_date = _parse_date(args.start_date, "start-date") if args.start_date else None
    end_date = _parse_date(args.end_date, "end-date") if args.end_date else None
    try:
        run_randomness_audit(
            run_date=run_date,
            input_path=args.input,
            output_dir=args.output_dir,
            start_date=start_date,
            end_date=end_date,
            skip_validation=args.skip_validation,
        )
    except Exception as exc:
        LOGGER.error("La auditoría falló: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
