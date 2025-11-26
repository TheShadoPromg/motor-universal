"""Panel Streamlit para visualizar predicciones y auditoría.

- Lee `predictions_daily` desde Postgres y permite explorar top-N, métricas y descargar CSV.
- Consume artefactos de auditoría de aleatoriedad para descarga/inspección (parquet->XLSX/ZIP/PDF).
- Sirve como UI operativa para validar runs diarios sin tocar el motor.
"""
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
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import textwrap

DB_URL = os.getenv(
    "PREDICTIONS_DB_URL",
    "postgresql+psycopg2://admin:admin@postgres:5432/motor",
)
engine = sa.create_engine(DB_URL, pool_pre_ping=True)

_audit_dir_env = os.getenv("AUDIT_RANDOMNESS_DIR")
if _audit_dir_env:
    AUDIT_BASE = Path(_audit_dir_env).expanduser().resolve()
else:
    AUDIT_BASE = (Path(__file__).resolve().parents[2] / "data" / "audit" / "randomness").resolve()

st.set_page_config(page_title="Panel Motor Universal", layout="wide")
st.title("Panel Motor Universal")
st.caption("Predicciones diarias y auditoría de aleatoriedad.")


def load_available_dates() -> List[pd.Timestamp]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT fecha FROM predictions_daily ORDER BY fecha DESC")
        )
        return [row[0] for row in rows]


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
    pattern = re.compile(r".*_(\d{4}-\d{2}-\d{2})\.parquet$")
    dates = set()
    for path in AUDIT_BASE.glob("frecuencia_global_resumen_*.parquet"):
        m = pattern.match(path.name)
        if m:
            dates.add(m.group(1))
    return sorted(dates, reverse=True)


def load_audit_table(run_date: str, stem: str) -> pd.DataFrame:
    path = AUDIT_BASE / f"{stem}_{run_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_all_audit_tables(run_date: str) -> dict:
    """Carga todas las tablas conocidas de auditoría para un run_date."""
    stems = [
        "frecuencia_global_numeros",
        "frecuencia_global_resumen",
        "frecuencia_por_posicion",
        "frecuencia_por_posicion_resumen",
        "par_impar_global",
        "par_impar_global_resumen",
        "par_impar_por_posicion",
        "par_impar_por_posicion_resumen",
        "alto_bajo_global",
        "alto_bajo_global_resumen",
        "alto_bajo_por_posicion",
        "alto_bajo_por_posicion_resumen",
        "decenas_global",
        "decenas_global_resumen",
        "decenas_por_posicion",
        "decenas_por_posicion_resumen",
        "repeticion_dias_consecutivos",
        "repeticion_dias_consecutivos_resumen",
        "condicional_reaparicion",
        "condicional_reaparicion_resumen",
        "rachas_par_impar",
        "rachas_par_impar_resumen",
        "rachas_alto_bajo",
        "rachas_alto_bajo_resumen",
        "rachas_repeticion",
        "rachas_repeticion_resumen",
    ]
    tables = {}
    for stem in stems:
        df = load_audit_table(run_date, stem)
        if not df.empty:
            tables[stem] = df
    return tables


tab_pred, tab_audit = st.tabs(["Predicciones", "Auditoría aleatoriedad"])

with tab_pred:
    available_dates = load_available_dates()
    if not available_dates:
        st.warning(
            "Aún no hay registros en la tabla `predictions_daily`. "
            "Ejecuta el pipeline diario para generar predicciones."
        )
    else:
        selected_date = st.selectbox("Fecha a visualizar", available_dates, format_func=str)
        top_n = st.slider("Top N a mostrar", min_value=10, max_value=100, value=20, step=5)

        predictions = load_predictions_for(selected_date)
        if predictions.empty:
            st.info("No se encontraron predicciones para la fecha seleccionada.")
        else:
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
    st.caption(f"Carpeta de auditoría: {AUDIT_BASE}")
    runs = _list_audit_runs()
    if not runs:
        st.info("No se encontraron artefactos en la carpeta de auditoría. Genera primero con engine.audit.randomness.")
    else:
        st.success(f"Runs disponibles: {', '.join(runs)}")
        run_date = st.selectbox("run_date disponible", runs)

        tables = _load_all_audit_tables(run_date)
        if tables:
            import io

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                for name, df in tables.items():
                    sheet_name = name[:31]  # Excel limita a 31 chars
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            buffer.seek(0)
            st.download_button(
                label="Descargar auditoría (Excel)",
                data=buffer,
                file_name=f"audit_randomness_{run_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            # Opción adicional: ZIP con cada tabla en CSV separado para consumo técnico.
            import zipfile

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, df in tables.items():
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    zf.writestr(f"{name}_{run_date}.csv", csv_bytes)
            zip_buffer.seek(0)
            st.download_button(
                label="Descargar tablas (ZIP con CSVs)",
                data=zip_buffer,
                file_name=f"audit_randomness_{run_date}.zip",
                mime="application/zip",
            )
            # PDF con todas las filas de cada tabla (puede ser grande; se asume que el usuario lo requiere completo).
            def _pdf_from_tables(tables_dict: dict) -> bytes:
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=10)
                pdf.add_page()
                pdf.set_font("Courier", size=10)
                pdf.cell(
                    0,
                    10,
                    f"Auditoria de Aleatoriedad - {run_date}",
                    new_x=XPos.LMARGIN,
                    new_y=YPos.NEXT,
                )
                pdf.set_font("Courier", size=8)
                for name, df in tables_dict.items():
                    pdf.ln(4)
                    pdf.set_font("Courier", "B", 9)
                    pdf.cell(
                        0,
                        8,
                        name,
                        new_x=XPos.LMARGIN,
                        new_y=YPos.NEXT,
                    )
                    pdf.set_font("Courier", size=6)
                    # Renderizamos la tabla en texto plano (todas las filas) formando líneas con separadores que permiten quiebres.
                    header_line = " , ".join(map(str, df.columns))
                    body_lines = [
                        " , ".join(map(str, row)) for row in df.to_records(index=False)
                    ]
                    for line in [header_line, *body_lines]:
                        pdf.set_x(pdf.l_margin)
                        # Evitamos errores de espacio horizontal partiendo líneas largas.
                        for chunk in textwrap.wrap(
                            line,
                            width=5,
                            break_long_words=True,
                            break_on_hyphens=True,
                        ):
                            pdf.cell(
                                pdf.epw,
                                4,
                                chunk,
                                new_x=XPos.LMARGIN,
                                new_y=YPos.NEXT,
                            )
                data = pdf.output(dest="S")
                # fpdf2 puede devolver bytearray; lo normalizamos a bytes.
                if isinstance(data, bytearray):
                    return bytes(data)
                if isinstance(data, str):
                    return data.encode("latin-1")
                return data

            pdf_bytes = _pdf_from_tables(tables)
            st.download_button(
                label="Descargar auditoría (PDF)",
                data=pdf_bytes,
                file_name=f"audit_randomness_{run_date}.pdf",
                mime="application/pdf",
            )
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
