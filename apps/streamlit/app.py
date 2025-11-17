from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
import streamlit as st

DB_URL = os.getenv(
    "PREDICTIONS_DB_URL",
    "postgresql+psycopg2://admin:admin@postgres:5432/motor",
)
engine = sa.create_engine(DB_URL, pool_pre_ping=True)

st.set_page_config(page_title="Predicciones Motor Universal", layout="wide")
st.title("Panel de Predicciones Diarias")
st.caption(
    "Explora las predicciones generadas por el pipeline y evalúa su calidad con los resultados reales."
)


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
