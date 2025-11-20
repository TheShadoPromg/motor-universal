from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import sqlalchemy as sa
from sqlalchemy import text
from pathlib import Path
import re

DB_URL = os.getenv(
    "PREDICTIONS_DB_URL",
    "postgresql+psycopg2://admin:admin@postgres:5432/motor",
)
engine = sa.create_engine(DB_URL, pool_pre_ping=True)
AUDIT_BASE = Path(os.getenv("AUDIT_RANDOMNESS_DIR", Path(__file__).resolve().parents[2] / "data" / "audit" / "randomness"))

st.set_page_config(page_title="Panel Motor Universal", layout="wide")
st.title("Panel Motor Universal")
st.caption("Predicciones diarias y auditoría de aleatoriedad.")


@st.cache_data(ttl=120)
def load_available_dates() -> List[pd.Timestamp]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT fecha FROM predictions_daily ORDER BY fecha DESC")
        )
        return [row[0] for row in rows]


@st.cache_data(ttl=60)
def load_predictions_for(date: pd.Timestamp) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT *
                FROM predictions_daily
                WHERE fecha = :fecha
                ORDER BY prob DESC, numero ASC
                """
            ),
            conn,
            params={"fecha": date},
        )
    df["prob"] = df["prob"].astype(float)
    df["prob_raw"] = df["prob_raw"].astype(float)
    return df


def _list_audit_runs() -> List[str]:
    if not AUDIT_BASE.exists():
        return []
    pattern = re.compile(r".*_(\\d{4}-\\d{2}-\\d{2})\\.parquet$")
    dates = set()
    for path in AUDIT_BASE.glob("frecuencia_global_resumen_*.parquet"):
        m = pattern.match(path.name)
        if m:
            dates.add(m.group(1))
    return sorted(dates, reverse=True)


@st.cache_data(ttl=60)
def load_audit_table(run_date: str, stem: str) -> pd.DataFrame:
    path = AUDIT_BASE / f"{stem}_{run_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


tab_pred, tab_audit = st.tabs(["Predicciones", "Auditoría aleatoriedad"])

with tab_pred:
    available_dates = load_available_dates()
    if not available_dates:
        st.warning(
            "Aún no hay registros en la tabla `predictions_daily`. "
            "Ejecuta el pipeline diario para generar predicciones."
        )
        st.stop()

    selected_date = st.selectbox("Fecha a visualizar", available_dates, format_func=str)
    top_n = st.slider("Top N a mostrar", min_value=10, max_value=100, value=20, step=5)

    predictions = load_predictions_for(selected_date)
    if predictions.empty:
        st.info("No se encontraron predicciones para la fecha seleccionada.")
        st.stop()

    left, mid, right = st.columns(3)
    left.metric(
        "Número top 1",
        f"{predictions.iloc[0]['numero']}",
        f"{predictions.iloc[0]['prob']:.2%}",
    )
    mid.metric(
        "Probabilidad acumulada (Top N)",
        f"{predictions.head(top_n)['prob'].sum():.2%}",
    )
    entropy = float(
        -(predictions["prob"] * np.log(predictions["prob"] + 1e-12)).sum()
    )
    right.metric("Entropía (menor=mejor)", f"{entropy:.3f}")

    st.subheader("Distribución de probabilidades")
    chart_df = predictions.head(top_n)[["numero", "prob"]].set_index("numero")
    st.bar_chart(chart_df)

    st.subheader(f"Detalle Top {top_n}")
    display_df = predictions.head(top_n).copy()
    display_df["prob"] = display_df["prob"].map(lambda x: f"{x:.2%}")
    display_df["prob_raw"] = display_df["prob_raw"].map(lambda x: f"{x:.2%}")
    st.dataframe(
        display_df[
            [
                "numero",
                "prob",
                "prob_raw",
                "score_cruzado",
                "score_estructural",
                "score_derivado",
                "score_total",
                "tipo_convergencia",
                "rank",
            ]
        ],
        hide_index=True,
    )

    csv = predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar predicciones (CSV)",
        data=csv,
        file_name=f"predictions_{selected_date}.csv",
        mime="text/csv",
    )

    st.subheader("Evaluación con resultados reales")
    actual_input = st.text_input(
        "Ingresa los números reales separados por coma (ej: 05,32,87)",
        help="Se comparan contra el ranking para mostrar la posición obtenida.",
    )
    actual_numbers = [item.strip() for item in actual_input.split(",") if item.strip()]
    if actual_numbers:
        matches = predictions[predictions["numero"].isin(actual_numbers)].copy()
        if matches.empty:
            st.error("Ninguno de los números ingresados apareció en las predicciones.")
        else:
            matches = matches.sort_values("rank")
            hits = len(matches)
            best_rank = int(matches["rank"].min())
            st.success(
                f"{hits} de {len(actual_numbers)} números aparecieron. "
                f"Mejor posición: #{best_rank}"
            )
            matches["prob"] = matches["prob"].map(lambda x: f"{x:.2%}")
            st.table(
                matches[["numero", "rank", "prob", "tipo_convergencia", "score_total"]].rename(
                    columns={"numero": "Número", "rank": "Posición", "prob": "Probabilidad"}
                )
            )

    st.caption(
        "La información proviene de la tabla `predictions_daily`. "
        "Actualiza la página para volver a consultar la base."
    )

with tab_audit:
    st.subheader("Auditoría de aleatoriedad (Fase 1)")
    runs = _list_audit_runs()
    if not runs:
        st.info("No se encontraron artefactos en la carpeta de auditoría. Genera primero con engine.audit.randomness.")
    else:
        run_date = st.selectbox("run_date disponible", runs)
        freq_summary = load_audit_table(run_date, "frecuencia_global_resumen")
        pos_summary = load_audit_table(run_date, "frecuencia_por_posicion_resumen")
        par_summary = load_audit_table(run_date, "par_impar_global_resumen")
        alto_summary = load_audit_table(run_date, "alto_bajo_global_resumen")
        dec_summary = load_audit_table(run_date, "decenas_global_resumen")
        repeticion_summary = load_audit_table(run_date, "repeticion_dias_consecutivos_resumen")
        cond_summary = load_audit_table(run_date, "condicional_reaparicion_resumen")

        col1, col2, col3 = st.columns(3)
        if not freq_summary.empty:
            col1.metric("Chi2 global números", f"{float(freq_summary.iloc[0]['chi2_global']):.3f}")
            pval = freq_summary.iloc[0]["p_value_global"]
            col2.metric("p-value global", f"{pval:.4f}" if pd.notna(pval) else "N/A")
            col3.metric("N sorteos", int(freq_summary.iloc[0]["N_sorteos"]))

        if not pos_summary.empty:
            st.markdown("**Chi2 por posición**")
            st.dataframe(pos_summary[["posicion", "chi2_pos", "p_value_pos", "desviacion_significativa_bool"]])

        st.markdown("**Categorías globales**")
        if not par_summary.empty or not alto_summary.empty or not dec_summary.empty:
            if not par_summary.empty:
                st.dataframe(par_summary)
            if not alto_summary.empty:
                st.dataframe(alto_summary)
            if not dec_summary.empty:
                st.dataframe(dec_summary)

        st.markdown("**Repetición y condicional**")
        if not repeticion_summary.empty:
            st.dataframe(repeticion_summary)
        if not cond_summary.empty:
            st.dataframe(cond_summary)

        with st.expander("Tablas completas"):
            freq_global = load_audit_table(run_date, "frecuencia_global_numeros")
            if not freq_global.empty:
                st.dataframe(freq_global, height=300)
            freq_pos = load_audit_table(run_date, "frecuencia_por_posicion")
            if not freq_pos.empty:
                st.dataframe(freq_pos, height=300)
            par_pos = load_audit_table(run_date, "par_impar_por_posicion")
            if not par_pos.empty:
                st.dataframe(par_pos, height=300)
            alto_pos = load_audit_table(run_date, "alto_bajo_por_posicion")
            if not alto_pos.empty:
                st.dataframe(alto_pos, height=300)
            dec_pos = load_audit_table(run_date, "decenas_por_posicion")
            if not dec_pos.empty:
                st.dataframe(dec_pos, height=300)
            rachas_par = load_audit_table(run_date, "rachas_par_impar")
            if not rachas_par.empty:
                st.dataframe(rachas_par, height=200)
            rachas_alto = load_audit_table(run_date, "rachas_alto_bajo")
            if not rachas_alto.empty:
                st.dataframe(rachas_alto, height=200)
            rachas_rep = load_audit_table(run_date, "rachas_repeticion")
            if not rachas_rep.empty:
                st.dataframe(rachas_rep, height=200)
