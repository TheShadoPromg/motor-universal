## Dashboard de Predicciones

El proyecto incluye un panel en Streamlit para inspeccionar los resultados diarios:

1. Levanta los servicios (incluido `streamlit`) con Docker:

   ```bash
   docker compose --env-file .env -f ops/docker/compose.yml up -d streamlit
   ```

2. Abre <http://localhost:8501>. Podrás:
   - Elegir la fecha procesada y ver el Top *N* con sus probabilidades.
   - Descargar el CSV completo.
   - Ingresar los números reales para evaluar rápidamente la posición que ocuparon.

El panel lee directamente de la tabla `predictions_daily`. Asegúrate de haber ejecutado el pipeline (`daily_pipeline`) para que existan registros.
