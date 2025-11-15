# Uso:
#   powershell -ExecutionPolicy Bypass -File .\env.ps1
# Activa el entorno virtual y exporta las variables necesarias
# para usar Prefect CLI apuntando al servidor local.

$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot"

# Activa el entorno virtual
.\.venv\Scripts\activate.ps1

# Prefect CLI debe apuntar al servidor local
$env:PREFECT_API_URL = "http://localhost:4200/api"
Write-Host "Prefect CLI apuntando a $env:PREFECT_API_URL"

# Opcional: recordatorio de credenciales S3/DB ya definidas en .env
Write-Host "Variables de entorno listas. Puedes ejecutar Prefect CLI desde esta consola."
