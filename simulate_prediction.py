import pandas as pd
import numpy as np
from datetime import timedelta

def run_simulation():
    print("Cargando datos...")
    
    # 1. Cargar Histórico (Base)
    try:
        df_draws = pd.read_parquet('data/base/eventos_numericos.parquet')
        df_draws['fecha'] = pd.to_datetime(df_draws['fecha'])
        df_draws['numero'] = df_draws['numero'].astype(int)
        last_date = df_draws['fecha'].max()
        print(f"Fecha de corte para predicción: {last_date.date()} (Prediciendo para el siguiente sorteo)")
    except Exception as e:
        print(f"Error cargando draws: {e}")
        return

    # 2. Cargar Reglas (Engine/Audit)
    try:
        df_hazard = pd.read_csv('engine/backtesting/hazard_numero_resumen.csv')
        # Limpiar y preparar hazard
        df_hazard['numero'] = pd.to_numeric(df_hazard['numero'], errors='coerce').fillna(-1).astype(int)
        
        df_struct = pd.read_csv('data/audit/estructural/sesgos_resumen_global_fase2.csv')
        df_struct['numero'] = pd.to_numeric(df_struct['numero'], errors='coerce').fillna(-1).astype(int)
    except Exception as e:
        print(f"Error cargando reglas: {e}")
        return

    # 3. Calcular Recencia Actual (Estado del Sistema)
    print("Calculando recencia actual...")
    recencia_map = {} # numero -> dias desde la ultima vez
    
    # Obtener la última fecha de aparición de cada número
    last_appearances = df_draws.groupby('numero')['fecha'].max()
    
    for num in range(100):
        if num in last_appearances:
            days_diff = (last_date - last_appearances[num]).days
            recencia_map[num] = days_diff
        else:
            recencia_map[num] = 999 # Nunca salió o hace mucho tiempo

    # 4. Motor de Predicción (Scoring)
    print("Ejecutando motor de inferencia...")
    predictions = []

    for num in range(100):
        recency = recencia_map[num]
        score = 0.0
        confidence = 0.0
        signals = []
        
        # --- REGLA A: Filtro "Valle de la Muerte" ---
        if 31 <= recency <= 45:
            score -= 50
            signals.append("FILTRO_RECENCIA_NEGATIVA")
        
        # --- REGLA B: Hazard (Recencia Corta) ---
        hazard_row = df_hazard[df_hazard['numero'] == num]
        if not hazard_row.empty:
            z_score = hazard_row.iloc[0]['z']
            p_val = hazard_row.iloc[0]['p_val']
            is_core = 'hazard_numero_core' in str(hazard_row.iloc[0]['clasificacion_hazard_numero'])
            
            if 1 <= recency <= 5:
                score += 10 # Base bonus por recencia corta
                
                if is_core:
                    score += 25
                    confidence += 0.3
                    signals.append(f"HAZARD_CORE (Z={z_score:.2f})")
                elif z_score > 1.5 and p_val < 0.1:
                    score += 10
                    confidence += 0.1
                    signals.append(f"HAZARD_SIGNAL (Z={z_score:.2f})")

        # --- REGLA C: Estructural (Resonancia de Lags) ---
        struct_rows = df_struct[df_struct['numero'] == num]
        for _, row in struct_rows.iterrows():
            lags_str = str(row['lags'])
            # Parsear lags "3,20" -> [3, 20]
            try:
                if ',' in lags_str:
                    target_lags = [int(x.strip()) for x in lags_str.split(',')]
                else:
                    target_lags = [int(lags_str)]
                
                if recency in target_lags:
                    delta = row['max_delta_rel']
                    boost = delta * 5 # Escalar el impacto
                    score += boost
                    confidence += (delta / 10)
                    signals.append(f"ESTRUCTURA_LAG (Lag={recency}, Delta={delta:.2f}x)")
            except:
                pass

        # --- Final Packaging ---
        if score > 0:
            predictions.append({
                'numero': f"{num:02d}",
                'recencia': recency,
                'score': round(score, 2),
                'confianza': round(min(confidence, 1.0) * 100, 1),
                'senales': ", ".join(signals)
            })

    # 5. Ranking y Salida
    df_pred = pd.DataFrame(predictions)
    if not df_pred.empty:
        df_pred = df_pred.sort_values('score', ascending=False).head(10)
        print("\n=== TOP PREDICCIONES PARA EL SIGUIENTE SORTEO ===")
        print(df_pred.to_string(index=False))
        
        # Escenarios
        print("\n=== ESCENARIOS DE VARIACIÓN ===")
        print("1. Conservador (Solo Hazard Core):")
        print(df_pred[df_pred['senales'].str.contains('HAZARD_CORE')].to_string(index=False))
        
        print("\n2. Especulativo (Estructura + Lags):")
        print(df_pred[df_pred['senales'].str.contains('ESTRUCTURA')].head(5).to_string(index=False))
    else:
        print("No se encontraron oportunidades positivas claras.")

if __name__ == "__main__":
    run_simulation()
