@echo off
cd /d %~dp0
echo Iniciando el servidor web PineDetect con Django...
python manage.py runserver 0.0.0.0:8000
pause
