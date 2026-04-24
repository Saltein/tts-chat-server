@echo off
cd /d %~dp0

call venv310\Scripts\activate.bat

python server.py

pause