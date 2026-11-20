param(
    [string]$PythonBase = 'python',
    [string]$Entorno = (Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer')
)
$ErrorActionPreference = 'Stop'
$raizProyecto = Split-Path -Parent $PSScriptRoot
$pythonEco = Join-Path $Entorno 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonEco)) {
    & $PythonBase -m venv $Entorno
    if ($LASTEXITCODE -ne 0) { throw 'No fue posible crear el entorno virtual.' }
}
& $pythonEco -m pip install -r (Join-Path $raizProyecto 'requirements-cuda.txt')
if ($LASTEXITCODE -ne 0) { throw 'Fallo al instalar PyTorch CUDA.' }
& $pythonEco -m pip install -r (Join-Path $raizProyecto 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Fallo al instalar las dependencias de imagen.' }
& $pythonEco (Join-Path $PSScriptRoot 'verificar_entorno.py') --salida (Join-Path $raizProyecto 'resultados\entorno.json')
if ($LASTEXITCODE -ne 0) { throw 'El diagnóstico del entorno detectó errores. Revisar el informe.' }
