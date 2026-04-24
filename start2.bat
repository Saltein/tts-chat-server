@echo off
cd /d %~dp0

echo Checking Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10 or higher.
    pause
    exit /b 1
)

:: Check if venv310 exists
if not exist "venv310\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv310
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    
    echo Installing dependencies...
    call venv310\Scripts\activate.bat
    
    :: Install from requirements.txt if exists
    if exist "requirements.txt" (
        pip install -r requirements.txt
    ) else (
        echo WARNING: requirements.txt not found
        echo Installing common packages (adjust as needed):
        pip install --upgrade pip
        :: Add your dependencies here, for example:
        :: pip install flask requests numpy
    )
    
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    call venv310\Scripts\activate.bat
)

echo Starting server...
python server.py

pause