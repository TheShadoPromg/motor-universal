from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine.audit import randomness

LOGGER = logging.getLogger("audit.estructural")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "eventos_numericos.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "audit" / "estructural"

CONFIG = {
    "L_min": 1,
    "L_max_core": 7,
    "L_max_extendido": 30,
    "N_min_oportunidades": 100,
    "alpha_bruto": 0.01,
    "delta_rel_min": 0.2,
    "segmentos_periodo": [
        ("2011_2014", date(2011, 1, 1), date(2014, 12, 31)),
        ("2015_2018", date(2015, 1, 1), date(2018, 12, 31)),
        ("2019_2022", date(2019, 1, 1), date(2022, 12, 31)),
        ("2023_2025", date(2023, 1, 1), date(2025, 12, 31)),
    ],
}

DAY_NAMES = {
    0: "LUNES",
    1: "MARTES",
    2: "MIERCOLES",
    3: "JUEVES",
    4: "VIERNES",
    5: "SABADO",
    6: "DOMINGO",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _parse_date(value: str, name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{name} debe tener formato YYYY-MM-DD") from exc


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _p_value_from_z(z: float) -> float:
    return 2 * (1 - _norm_cdf(abs(z)))


def _determine_period(fecha: date) -> Optional[str]:
    for label, start, end in CONFIG["segmentos_periodo"]:
        if start <= fecha <= end:
            return label
    return None


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["anio"] = df["fecha"].apply(lambda d: d.year)
    df["mes"] = df["fecha"].apply(lambda d: d.month)
    df["dia_semana_idx"] = df["fecha"].apply(lambda d: d.weekday())
    df["segmento_dia_semana"] = df["dia_semana_idx"].map(DAY_NAMES)
    df["es_fin_de_semana"] = df["dia_semana_idx"].isin([5, 6])
    df["paridad"] = np.where(df["numero"] % 2 == 0, "par", "impar")
    df["rango_50"] = np.where(df["numero"] >= 50, "alto", "bajo")
    df["decena"] = df["numero"] // 10
    df["unidad"] = df["numero"] % 10
    df["espejo_99"] = 99 - df["numero"]
    df["consecutivo_p1"] = (df["numero"] + 1) % 100
    df["consecutivo_m1"] = (df["numero"] - 1) % 100
    df["segmento_periodo"] = df["fecha"].apply(_determine_period)
    df["segmento_compuesto"] = df.apply(
        lambda row: f"{row['segmento_periodo']}_{row['segmento_dia_semana']}" if pd.notna(row["segmento_periodo"]) else pd.NA,
        axis=1,
    )
    return df


def _add_run_date(df: pd.DataFrame, run_date: date) -> pd.DataFrame:
    df = df.copy()
    df["run_date"] = run_date.strftime("%Y-%m-%d")
    return df


def _build_daily_lookup(df: pd.DataFrame) -> Tuple[List[date], Dict[date, int], List[set], List[Dict[int, int]]]:
    grouped = df.groupby("fecha")
    dates = sorted(grouped.groups.keys())
    date_index = {d: idx for idx, d in enumerate(dates)}
    numbers_by_date = []
    positions_by_date = []
    for d in dates:
        subset = grouped.get_group(d)
        numbers_by_date.append(set(subset["numero"].tolist()))
        positions_by_date.append({int(row["posicion"]): int(row["numero"]) for _, row in subset.iterrows()})
    return dates, date_index, numbers_by_date, positions_by_date


def _compute_metrics(n_opp: int, n_success: int, p_teorica: float) -> Tuple[float, float, float, float, float]:
    if n_opp <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    p_emp = n_success / n_opp
    delta_abs = p_emp - p_teorica
    delta_rel = delta_abs / p_teorica if p_teorica > 0 else float("nan")
    std = math.sqrt(p_teorica * (1 - p_teorica) / n_opp) if p_teorica > 0 else float("nan")
    z_score = delta_abs / std if std and std > 0 else float("nan")
    p_value = _p_value_from_z(z_score) if not math.isnan(z_score) else float("nan")
    return p_emp, delta_abs, delta_rel, z_score, p_value


def _is_bias(n_opp: int, delta_rel: float, p_value: float) -> bool:
    if n_opp < CONFIG["N_min_oportunidades"]:
        return False
    if math.isnan(delta_rel) or math.isnan(p_value):
        return False
    return abs(delta_rel) >= CONFIG["delta_rel_min"] and p_value <= CONFIG["alpha_bruto"]


def _iter_segments(df: pd.DataFrame, include_compuestos: bool) -> Iterable[Tuple[str, str, pd.DataFrame]]:
    yield "GLOBAL", "GLOBAL", df

    for period in sorted([p for p in df["segmento_periodo"].dropna().unique()]):
        subset = df[df["segmento_periodo"] == period]
        if not subset.empty:
            yield "PERIODO", str(period), subset

    for day in DAY_NAMES.values():
        subset = df[df["segmento_dia_semana"] == day]
        if not subset.empty:
            yield "DIA_SEMANA", day, subset

    if include_compuestos:
        for comp in sorted([c for c in df["segmento_compuesto"].dropna().unique()]):
            subset = df[df["segmento_compuesto"] == comp]
            if not subset.empty:
                yield "COMPUESTO", str(comp), subset


def _transiciones_numeros(
    dates: List[date],
    date_index: Dict[date, int],
    numbers_by_date: List[set],
    positions_by_date: List[Dict[int, int]],
    lags: Sequence[int],
    segment_type: str,
    segment_value: str,
) -> pd.DataFrame:
    oportunidades = defaultdict(int)
    exitos_any = defaultdict(int)
    exitos_pos = defaultdict(int)

    for idx, current_date in enumerate(dates):
        pos_today = positions_by_date[idx]
        for lag in lags:
            target_date = current_date + timedelta(days=lag)
            target_idx = date_index.get(target_date)
            if target_idx is None:
                continue
            target_numbers = numbers_by_date[target_idx]
            target_positions = positions_by_date[target_idx]
            for pos_origen, numero in pos_today.items():
                oportunidades[(numero, 0, lag)] += 1
                oportunidades[(numero, pos_origen, lag)] += 1
                if numero in target_numbers:
                    exitos_any[(numero, 0, lag)] += 1
                    exitos_any[(numero, pos_origen, lag)] += 1
                for pos_destino, numero_destino in target_positions.items():
                    if numero_destino == numero:
                        exitos_pos[(numero, 0, pos_destino, lag)] += 1
                        exitos_pos[(numero, pos_origen, pos_destino, lag)] += 1

    rows = []
    for numero in range(100):
        for lag in lags:
            n_opp_any = oportunidades.get((numero, 0, lag), 0)
            n_success_any = exitos_any.get((numero, 0, lag), 0)
            p_emp, delta_abs, delta_rel, z_score, p_value = _compute_metrics(n_opp_any, n_success_any, p_teorica=0.03)
            rows.append(
                {
                    "tipo_relacion": "numero",
                    "numero": f"{numero:02d}",
                    "numero_objetivo": f"{numero:02d}",
                    "pos_origen": "ANY",
                    "pos_destino": "ANY",
                    "lag": lag,
                    "segmento_tipo": segment_type,
                    "segmento_valor": segment_value,
                    "n_oportunidades": n_opp_any,
                    "n_exitos": n_success_any,
                    "p_empirica": p_emp,
                    "p_teorica": 0.03,
                    "delta_abs": delta_abs,
                    "delta_rel": delta_rel,
                    "z_score": z_score,
                    "p_value": p_value,
                    "es_sesgo_significativo": _is_bias(n_opp_any, delta_rel, p_value),
                    "nota": "",
                }
            )
            for pos_destino in (1, 2, 3):
                n_success_pos = exitos_pos.get((numero, 0, pos_destino, lag), 0)
                p_emp, delta_abs, delta_rel, z_score, p_value = _compute_metrics(n_opp_any, n_success_pos, p_teorica=0.01)
                rows.append(
                    {
                        "tipo_relacion": "numero",
                        "numero": f"{numero:02d}",
                        "numero_objetivo": f"{numero:02d}",
                        "pos_origen": "ANY",
                        "pos_destino": pos_destino,
                        "lag": lag,
                        "segmento_tipo": segment_type,
                        "segmento_valor": segment_value,
                        "n_oportunidades": n_opp_any,
                        "n_exitos": n_success_pos,
                        "p_empirica": p_emp,
                        "p_teorica": 0.01,
                        "delta_abs": delta_abs,
                        "delta_rel": delta_rel,
                        "z_score": z_score,
                        "p_value": p_value,
                        "es_sesgo_significativo": _is_bias(n_opp_any, delta_rel, p_value),
                        "nota": "",
                    }
                )

            for pos_origen in (1, 2, 3):
                n_opp_pos = oportunidades.get((numero, pos_origen, lag), 0)
                n_success_any_pos = exitos_any.get((numero, pos_origen, lag), 0)
                p_emp, delta_abs, delta_rel, z_score, p_value = _compute_metrics(n_opp_pos, n_success_any_pos, p_teorica=0.03)
                rows.append(
                    {
                        "tipo_relacion": "numero",
                        "numero": f"{numero:02d}",
                        "numero_objetivo": f"{numero:02d}",
                        "pos_origen": pos_origen,
                        "pos_destino": "ANY",
                        "lag": lag,
                        "segmento_tipo": segment_type,
                        "segmento_valor": segment_value,
                        "n_oportunidades": n_opp_pos,
                        "n_exitos": n_success_any_pos,
                        "p_empirica": p_emp,
                        "p_teorica": 0.03,
                        "delta_abs": delta_abs,
                        "delta_rel": delta_rel,
                        "z_score": z_score,
                        "p_value": p_value,
                        "es_sesgo_significativo": _is_bias(n_opp_pos, delta_rel, p_value),
                        "nota": "",
                    }
                )
                for pos_destino in (1, 2, 3):
                    n_success = exitos_pos.get((numero, pos_origen, pos_destino, lag), 0)
                    p_emp, delta_abs, delta_rel, z_score, p_value = _compute_metrics(n_opp_pos, n_success, p_teorica=0.01)
                    rows.append(
                        {
                            "tipo_relacion": "numero",
                            "numero": f"{numero:02d}",
                            "numero_objetivo": f"{numero:02d}",
                            "pos_origen": pos_origen,
                            "pos_destino": pos_destino,
                            "lag": lag,
                            "segmento_tipo": segment_type,
                            "segmento_valor": segment_value,
                            "n_oportunidades": n_opp_pos,
                            "n_exitos": n_success,
                            "p_empirica": p_emp,
                            "p_teorica": 0.01,
                            "delta_abs": delta_abs,
                            "delta_rel": delta_rel,
                            "z_score": z_score,
                            "p_value": p_value,
                            "es_sesgo_significativo": _is_bias(n_opp_pos, delta_rel, p_value),
                            "nota": "",
                        }
                    )
    return pd.DataFrame(rows)


def _transiciones_relacion(
    dates: List[date],
    date_index: Dict[date, int],
    positions_by_date: List[Dict[int, int]],
    lags: Sequence[int],
    segment_type: str,
    segment_value: str,
    relation_name: str,
    target_func,
) -> pd.DataFrame:
    oportunidades = defaultdict(int)
    exitos_any = defaultdict(int)
    exitos_pos = defaultdict(int)

    for idx, current_date in enumerate(dates):
        pos_today = positions_by_date[idx]
        for lag in lags:
            target_date = current_date + timedelta(days=lag)
            target_idx = date_index.get(target_date)
            if target_idx is None:
                continue
            target_positions = positions_by_date[target_idx]
            target_numbers_set = set(target_positions.values())
            for pos_origen, numero in pos_today.items():
                objetivo = target_func(numero)
                oportunidades[(numero, pos_origen, lag)] += 1
                oportunidades[(numero, 0, lag)] += 1
                if objetivo in target_numbers_set:
                    exitos_any[(numero, pos_origen, lag)] += 1
                    exitos_any[(numero, 0, lag)] += 1
                for pos_destino, numero_destino in target_positions.items():
                    if numero_destino == objetivo:
                        exitos_pos[(numero, pos_origen, pos_destino, lag)] += 1
                        exitos_pos[(numero, 0, pos_destino, lag)] += 1

    rows = []
    for numero in range(100):
        for lag in lags:
            objetivo = target_func(numero)
            n_opp_any = oportunidades.get((numero, 0, lag), 0)
            n_success_any = exitos_any.get((numero, 0, lag), 0)
            p_emp, delta_abs, delta_rel, z_score, p_value = _compute_metrics(n_opp_any, n_success_any, p_teorica=0.03)
            rows.append(
                {
                    "tipo_relacion": relation_name,
                    "numero": f"{numero:02d}",
                    "numero_objetivo": f"{objetivo:02d}",
                    "pos_origen": "ANY",
                    "pos_destino": "ANY",
                    "lag": lag,
                    "segmento_tipo": segment_type,
                    "segmento_valor": segment_value,
                    "n_oportunidades": n_opp_any,
                    "n_exitos": n_success_any,
                    "p_empirica": p_emp,
                    "p_teorica": 0.03,
                    "delta_abs": delta_abs,
                    "delta_rel": delta_rel,
                    "z_score": z_score,
                    "p_value": p_value,
                    "es_sesgo_significativo": _is_bias(n_opp_any, delta_rel, p_value),
                    "nota": "",
                }
            )

            for pos_origen in (1, 2, 3):
                n_opp_pos = oportunidades.get((numero, pos_origen, lag), 0)
                n_success_any_pos = exitos_any.get((numero, pos_origen, lag), 0)
                p_emp, delta_abs, delta_rel, z_score, p_value = _compute_metrics(n_opp_pos, n_success_any_pos, p_teorica=0.03)
                rows.append(
                    {
                        "tipo_relacion": relation_name,
                        "numero": f"{numero:02d}",
                        "numero_objetivo": f"{objetivo:02d}",
                        "pos_origen": pos_origen,
                        "pos_destino": "ANY",
                        "lag": lag,
                        "segmento_tipo": segment_type,
                        "segmento_valor": segment_value,
                        "n_oportunidades": n_opp_pos,
                        "n_exitos": n_success_any_pos,
                        "p_empirica": p_emp,
                        "p_teorica": 0.03,
                        "delta_abs": delta_abs,
                        "delta_rel": delta_rel,
                        "z_score": z_score,
                        "p_value": p_value,
                        "es_sesgo_significativo": _is_bias(n_opp_pos, delta_rel, p_value),
                        "nota": "",
                    }
                )
                for pos_destino in (1, 2, 3):
                    n_success = exitos_pos.get((numero, pos_origen, pos_destino, lag), 0)
                    p_emp, delta_abs, delta_rel, z_score, p_value = _compute_metrics(n_opp_pos, n_success, p_teorica=0.01)
                    rows.append(
                        {
                            "tipo_relacion": relation_name,
                            "numero": f"{numero:02d}",
                            "numero_objetivo": f"{objetivo:02d}",
                            "pos_origen": pos_origen,
                            "pos_destino": pos_destino,
                            "lag": lag,
                            "segmento_tipo": segment_type,
                            "segmento_valor": segment_value,
                            "n_oportunidades": n_opp_pos,
                            "n_exitos": n_success,
                            "p_empirica": p_emp,
                            "p_teorica": 0.01,
                            "delta_abs": delta_abs,
                            "delta_rel": delta_rel,
                            "z_score": z_score,
                            "p_value": p_value,
                            "es_sesgo_significativo": _is_bias(n_opp_pos, delta_rel, p_value),
                            "nota": "",
                        }
                    )
    return pd.DataFrame(rows)


def _p_at_least_one(size: int, universe: int = 100, draws: int = 3) -> float:
    if size <= 0 or draws <= 0 or universe <= 0:
        return 0.0
    if size >= universe:
        return 1.0
    from math import comb

    none = comb(universe - size, draws) / comb(universe, draws)
    return 1 - none


def _transiciones_categorias(
    dates: List[date],
    date_index: Dict[date, int],
    positions_by_date: List[Dict[int, int]],
    lags: Sequence[int],
    segment_type: str,
    segment_value: str,
) -> pd.DataFrame:
    categorias = {
        "paridad": (lambda n: "par" if n % 2 == 0 else "impar", {"par": 50, "impar": 50}),
        "rango_50": (lambda n: "alto" if n >= 50 else "bajo", {"alto": 50, "bajo": 50}),
        "decena": (lambda n: str(n // 10), {str(k): 10 for k in range(10)}),
        "unidad": (lambda n: str(n % 10), {str(k): 10 for k in range(10)}),
    }

    rows = []
    for cat_name, (func, sizes) in categorias.items():
        oportunidades = defaultdict(int)
        exitos = defaultdict(int)
        for idx, current_date in enumerate(dates):
            pos_today = positions_by_date[idx]
            for lag in lags:
                target_date = current_date + timedelta(days=lag)
                target_idx = date_index.get(target_date)
                if target_idx is None:
                    continue
                target_positions = positions_by_date[target_idx]
                target_cats = {pos: func(num) for pos, num in target_positions.items()}
                for _, numero in pos_today.items():
                    origen = func(numero)
                    oportunidades[(origen, "ANY", lag)] += 1
                    for pos_dest, dest_cat in target_cats.items():
                        exitos[(origen, dest_cat, "ANY", lag)] += 1
                        exitos[(origen, dest_cat, pos_dest, lag)] += 1
        for origen_label, size in sizes.items():
            p_teorica_any = _p_at_least_one(size)
            p_teorica_pos = size / 100.0
            for destino_label in sizes.keys():
                for lag in lags:
                    n_opp = oportunidades.get((origen_label, "ANY", lag), 0)
                    n_success_any = exitos.get((origen_label, destino_label, "ANY", lag), 0)
                    p_emp, delta_abs, delta_rel, z_score, p_value = _compute_metrics(n_opp, n_success_any, p_teorica_any)
                    rows.append(
                        {
                            "tipo_relacion": f"categoria_{cat_name}",
                            "categoria_origen": origen_label,
                            "categoria_destino": destino_label,
                            "pos_destino": "ANY",
                            "lag": lag,
                            "segmento_tipo": segment_type,
                            "segmento_valor": segment_value,
                            "n_oportunidades": n_opp,
                            "n_exitos": n_success_any,
                            "p_empirica": p_emp,
                            "p_teorica": p_teorica_any,
                            "delta_abs": delta_abs,
                            "delta_rel": delta_rel,
                            "z_score": z_score,
                            "p_value": p_value,
                            "es_sesgo_significativo": _is_bias(n_opp, delta_rel, p_value),
                            "nota": "",
                        }
                    )
                    for pos_dest in (1, 2, 3):
                        n_success_pos = exitos.get((origen_label, destino_label, pos_dest, lag), 0)
                        p_emp, delta_abs, delta_rel, z_score, p_value = _compute_metrics(n_opp, n_success_pos, p_teorica_pos)
                        rows.append(
                            {
                                "tipo_relacion": f"categoria_{cat_name}",
                                "categoria_origen": origen_label,
                                "categoria_destino": destino_label,
                                "pos_destino": pos_dest,
                                "lag": lag,
                                "segmento_tipo": segment_type,
                                "segmento_valor": segment_value,
                                "n_oportunidades": n_opp,
                                "n_exitos": n_success_pos,
                                "p_empirica": p_emp,
                                "p_teorica": p_teorica_pos,
                                "delta_abs": delta_abs,
                                "delta_rel": delta_rel,
                                "z_score": z_score,
                                "p_value": p_value,
                                "es_sesgo_significativo": _is_bias(n_opp, delta_rel, p_value),
                                "nota": "",
                            }
                        )
    return pd.DataFrame(rows)


def _intradiario_sumas(df: pd.DataFrame, run_date: date) -> pd.DataFrame:
    pair_labels = {(1, 2): "1_2", (1, 3): "1_3", (2, 3): "2_3"}
    rows = []
    grouped = df.groupby("fecha")
    for fecha, subset in grouped:
        nums = {int(row["posicion"]): int(row["numero"]) for _, row in subset.iterrows()}
        for (p1, p2), label in pair_labels.items():
            suma = nums[p1] + nums[p2]
            rows.append({"fecha": fecha, "pair": label, "suma_mod_10": suma % 10, "suma_mod_100": suma % 100})

    result_rows = []
    for label in pair_labels.values():
        pair_df = pd.DataFrame([r for r in rows if r["pair"] == label])
        total = len(pair_df)
        for modulo, max_val in (("suma_mod_10", 10), ("suma_mod_100", 100)):
            counts = pair_df[modulo].value_counts().reindex(range(max_val), fill_value=0).sort_index()
            esperado = total / max_val if total > 0 else 0.0
            for valor, cnt in counts.items():
                freq = cnt / total if total > 0 else 0.0
                delta_abs = freq - (1 / max_val if max_val > 0 else 0.0)
                delta_rel = delta_abs / (1 / max_val) if max_val > 0 else float("nan")
                result_rows.append(
                    {
                        "pair": label,
                        "modulo": modulo,
                        "valor": int(valor),
                        "count": int(cnt),
                        "total": int(total),
                        "freq": freq,
                        "esperado": esperado,
                        "delta_abs": delta_abs,
                        "delta_rel": delta_rel,
                        "run_date": run_date.strftime("%Y-%m-%d"),
                    }
                )
    return pd.DataFrame(result_rows)


def _sesgos_por_dia(transiciones: pd.DataFrame) -> pd.DataFrame:
    subset = transiciones[transiciones["segmento_tipo"] == "DIA_SEMANA"]
    if subset.empty:
        return pd.DataFrame(columns=["segmento_valor", "tipo_relacion", "n_sesgos"])
    grouped = subset.groupby(["segmento_valor", "tipo_relacion"])["es_sesgo_significativo"].sum().reset_index()
    grouped = grouped.rename(columns={"es_sesgo_significativo": "n_sesgos"})
    return grouped


def _resumen_global(bases: List[pd.DataFrame]) -> pd.DataFrame:
    all_rows = pd.concat(bases, ignore_index=True) if bases else pd.DataFrame()
    if all_rows.empty:
        return pd.DataFrame(
            columns=["numero", "tipo_relacion", "max_delta_rel", "segmentos", "lags", "clasificacion", "signo"]
        )
    sesgos = all_rows[all_rows["es_sesgo_significativo"]].copy()
    if sesgos.empty:
        return pd.DataFrame(
            columns=["numero", "tipo_relacion", "max_delta_rel", "segmentos", "lags", "clasificacion", "signo"]
        )

    def _clasificar(group: pd.DataFrame) -> str:
        segmentos = group["segmento_tipo"].nunique()
        if "GLOBAL" in group["segmento_tipo"].unique():
            return "estructura_estable_global"
        if segmentos >= 2:
            return "estructura_periodica"
        return "ruido"

    resumen_rows = []
    for (numero, tipo_relacion), grp in sesgos.groupby(["numero", "tipo_relacion"]):
        max_delta_rel = float(grp["delta_rel"].abs().max())
        segmentos = sorted(grp["segmento_valor"].unique())
        lags = sorted(grp["lag"].unique())
        signo = "positivo" if grp["delta_rel"].mean() >= 0 else "negativo"
        clasificacion = _clasificar(grp)
        resumen_rows.append(
            {
                "numero": numero,
                "tipo_relacion": tipo_relacion,
                "max_delta_rel": max_delta_rel,
                "segmentos": ",".join(segmentos),
                "lags": ",".join(str(l) for l in lags),
                "clasificacion": clasificacion,
                "signo": signo,
            }
        )
    return pd.DataFrame(resumen_rows)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_structural_audit(
    run_date: date,
    input_path: str,
    output_dir: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    skip_validation: bool = False,
    incluir_lags_ext: bool = False,
    incluir_compuestos: bool = False,
) -> Dict[str, pd.DataFrame]:
    LOGGER.info("Iniciando auditoria estructural (Fase 2) para run_date=%s", run_date)
    raw_df = randomness._read_events(Path(input_path))
    events = randomness._normalize_events(raw_df, start_date, end_date)
    if len(events) == 0:
        raise ValueError("El dataset de eventos esta vacio despues de filtrar por fechas.")

    events = _add_derived_columns(events)
    lags_max = CONFIG["L_max_extendido"] if incluir_lags_ext else CONFIG["L_max_core"]
    lags = list(range(CONFIG["L_min"], lags_max + 1))

    trans_num_list: List[pd.DataFrame] = []
    trans_espejo_list: List[pd.DataFrame] = []
    trans_consec_list: List[pd.DataFrame] = []
    trans_categorias_list: List[pd.DataFrame] = []

    for segmento_tipo, segmento_valor, subset in _iter_segments(events, include_compuestos=incluir_compuestos):
        dates, date_index, numbers_by_date, positions_by_date = _build_daily_lookup(subset)
        if not dates:
            continue
        trans_num_list.append(
            _transiciones_numeros(
                dates=dates,
                date_index=date_index,
                numbers_by_date=numbers_by_date,
                positions_by_date=positions_by_date,
                lags=lags,
                segment_type=segmento_tipo,
                segment_value=segmento_valor,
            )
        )
        trans_espejo_list.append(
            _transiciones_relacion(
                dates=dates,
                date_index=date_index,
                positions_by_date=positions_by_date,
                lags=lags,
                segment_type=segmento_tipo,
                segment_value=segmento_valor,
                relation_name="espejo",
                target_func=lambda n: 99 - n,
            )
        )
        trans_consec_list.append(
            _transiciones_relacion(
                dates=dates,
                date_index=date_index,
                positions_by_date=positions_by_date,
                lags=lags,
                segment_type=segmento_tipo,
                segment_value=segmento_valor,
                relation_name="consecutivo_+1",
                target_func=lambda n: (n + 1) % 100,
            )
        )
        trans_consec_list.append(
            _transiciones_relacion(
                dates=dates,
                date_index=date_index,
                positions_by_date=positions_by_date,
                lags=lags,
                segment_type=segmento_tipo,
                segment_value=segmento_valor,
                relation_name="consecutivo_-1",
                target_func=lambda n: (n - 1) % 100,
            )
        )
        trans_categorias_list.append(
            _transiciones_categorias(
                dates=dates,
                date_index=date_index,
                positions_by_date=positions_by_date,
                lags=lags,
                segment_type=segmento_tipo,
                segment_value=segmento_valor,
            )
        )

    transiciones_numero = pd.concat(trans_num_list, ignore_index=True) if trans_num_list else pd.DataFrame()
    transiciones_espejo = pd.concat(trans_espejo_list, ignore_index=True) if trans_espejo_list else pd.DataFrame()
    transiciones_consecutivo = (
        pd.concat(trans_consec_list, ignore_index=True) if trans_consec_list else pd.DataFrame()
    )
    transiciones_categorias = (
        pd.concat(trans_categorias_list, ignore_index=True) if trans_categorias_list else pd.DataFrame()
    )

    intradiario = _intradiario_sumas(events, run_date)
    sesgos_dia = _sesgos_por_dia(transiciones_numero)
    resumen_global = _resumen_global([transiciones_numero, transiciones_espejo, transiciones_consecutivo])

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    _save_csv(
        transiciones_numero[
            (transiciones_numero["pos_origen"] == "ANY") & (transiciones_numero["pos_destino"] == "ANY")
        ],
        output_base / "transiciones_numero_anypos_anypos.csv",
    )
    _save_csv(
        transiciones_numero[
            (transiciones_numero["pos_origen"] == "ANY") & (transiciones_numero["pos_destino"].isin([1, 2, 3]))
        ],
        output_base / "transiciones_numero_anypos_pos.csv",
    )
    _save_csv(
        transiciones_numero[
            (transiciones_numero["pos_origen"].isin([1, 2, 3])) & (transiciones_numero["pos_destino"].isin([1, 2, 3]))
        ],
        output_base / "transiciones_numero_pos_pos.csv",
    )

    _save_csv(
        transiciones_espejo[
            (transiciones_espejo["pos_origen"] == "ANY") & (transiciones_espejo["pos_destino"] == "ANY")
        ],
        output_base / "transiciones_espejo_anypos_anypos.csv",
    )
    _save_csv(
        transiciones_espejo[
            (transiciones_espejo["pos_origen"].isin([1, 2, 3])) & (transiciones_espejo["pos_destino"].isin([1, 2, 3]))
        ],
        output_base / "transiciones_espejo_pos_pos.csv",
    )

    _save_csv(
        transiciones_consecutivo[
            (transiciones_consecutivo["pos_origen"] == "ANY") & (transiciones_consecutivo["pos_destino"] == "ANY")
        ],
        output_base / "transiciones_consecutivo_anypos_anypos.csv",
    )
    _save_csv(
        transiciones_consecutivo[
            (transiciones_consecutivo["pos_origen"].isin([1, 2, 3])) & (transiciones_consecutivo["pos_destino"].isin([1, 2, 3]))
        ],
        output_base / "transiciones_consecutivo_pos_pos.csv",
    )

    _save_csv(transiciones_categorias, output_base / "transiciones_categorias.csv")
    _save_csv(intradiario, output_base / "estructuras_intradiarias_sumas_resumen.csv")
    _save_csv(sesgos_dia, output_base / "sesgos_por_dia_semana_resumen.csv")
    _save_csv(resumen_global, output_base / "sesgos_resumen_global_fase2.csv")

    LOGGER.info("Auditoria estructural completada. Artefactos guardados en %s", output_base)

    return {
        "transiciones_numero": transiciones_numero,
        "transiciones_espejo": transiciones_espejo,
        "transiciones_consecutivo": transiciones_consecutivo,
        "transiciones_categorias": transiciones_categorias,
        "intradiario": intradiario,
        "sesgos_por_dia": sesgos_dia,
        "resumen_global": resumen_global,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoria Fase 2 de sesgos estructurales sobre eventos_numericos.")
    parser.add_argument("--run-date", required=False, help="Fecha logica de ejecucion (YYYY-MM-DD).")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Ruta al CSV/Parquet de eventos_numericos.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio base para artefactos CSV.")
    parser.add_argument("--start-date", default=None, help="Fecha inicial a incluir (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Fecha final a incluir (YYYY-MM-DD).")
    parser.add_argument("--skip-validation", action="store_true", help="Omite validaciones externas adicionales.")
    parser.add_argument(
        "--include-extended-lags",
        action="store_true",
        help=f"Incluye lags hasta L_max_extendido={CONFIG['L_max_extendido']} ademas del rango core.",
    )
    parser.add_argument(
        "--include-compuestos",
        action="store_true",
        help="Habilita el calculo de segmentos compuestos periodo + dia de la semana.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)
    run_date = _parse_date(args.run_date, "run-date") if args.run_date else datetime.utcnow().date()
    start_date = _parse_date(args.start_date, "start-date") if args.start_date else None
    end_date = _parse_date(args.end_date, "end-date") if args.end_date else None
    try:
        run_structural_audit(
            run_date=run_date,
            input_path=args.input,
            output_dir=args.output_dir,
            start_date=start_date,
            end_date=end_date,
            skip_validation=args.skip_validation,
            incluir_lags_ext=args.include_extended_lags,
            incluir_compuestos=args.include_compuestos,
        )
    except Exception as exc:
        LOGGER.error("La auditoria estructural fallo: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
