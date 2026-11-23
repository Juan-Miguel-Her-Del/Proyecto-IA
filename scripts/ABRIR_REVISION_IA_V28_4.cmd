@echo off
cd /d "%~dp0.."
"%USERPROFILE%\.venvs\EcoWasteAnotacion\Scripts\python.exe" scripts/iniciar_editor_v28_4.py
if errorlevel 1 goto error
start "" http://127.0.0.1:8771/
exit /b 0
:error
echo Revisa el mensaje anterior. No se sustituyo ningun proyecto.
pause
