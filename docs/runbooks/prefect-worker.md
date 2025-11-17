# Prefect Worker en Docker

1. Asegurate de tener `.env` completo en la raiz del repo (se carga dentro del contenedor) y construye la imagen del worker:  
   `docker compose -f ops/docker/compose.yml build worker`
2. Levanta toda la pila (o solo el worker junto al servidor Prefect) con:  
   `docker compose -f ops/docker/compose.yml up -d prefect worker`
3. Verifica los logs del worker para confirmar que se conecto al pool configurado y heredo las credenciales AWS/DB:  
   `docker compose -f ops/docker/compose.yml logs -f worker`
4. Cada vez que cambies codigo o dependencias vuelve a registrar el deployment:  
   `python flows/register_daily_flow.py`
5. Los jobs programados apareceran automaticamente en `http://localhost:4200`. Si necesitas disparar uno manualmente usa:  
   `prefect deployment run daily_pipeline/daily-pipeline --params '{"run_date":"YYYY-MM-DD"}'`
