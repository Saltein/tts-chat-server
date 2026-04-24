@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   Python 3.10 Installer + Requirements + Server
echo ===================================================
echo.

:: Проверяем, запущен ли скрипт с правами администратора
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARN] Этот скрипт требует права администратора для установки Python.
    echo Пожалуйста, запустите файл от имени администратора.
    echo.
    pause
    exit /b 1
)

:: Определяем папку для установки Python
set "PYTHON_INSTALL_DIR=C:\Python310"
set "PYTHON_EXE=%PYTHON_INSTALL_DIR%\python.exe"

:: Шаг 1: Проверяем, установлен ли Python 3.10
echo [1/4] Проверка наличия Python 3.10...
%PYTHON_EXE% --version 2>nul | find "3.10" >nul
if %errorLevel% equ 0 (
    echo [OK] Python 3.10 уже установлен в %PYTHON_INSTALL_DIR%
) else (
    echo [INFO] Python 3.10 не найден. Начинаем загрузку...
    
    :: Скачиваем Python 3.10 установщик
    set "INSTALLER=python-3.10.11-amd64.exe"
    set "DOWNLOAD_URL=https://www.python.org/ftp/python/3.10.11/%INSTALLER%"
    
    echo Скачивание %DOWNLOAD_URL% ...
    powershell -Command "Invoke-WebRequest -Uri %DOWNLOAD_URL% -OutFile %INSTALLER%"
    if %errorLevel% neq 0 (
        echo [ERROR] Не удалось скачать Python.
        pause
        exit /b 1
    )
    echo [OK] Скачивание завершено.
    
    echo Установка Python 3.10 в %PYTHON_INSTALL_DIR%...
    %INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 TargetDir=%PYTHON_INSTALL_DIR%
    if %errorLevel% neq 0 (
        echo [ERROR] Ошибка при установке Python.
        del %INSTALLER%
        pause
        exit /b 1
    )
    
    del %INSTALLER%
    echo [OK] Python 3.10 успешно установлен.
)

:: Шаг 2: Обновляем PATH (на всякий случай)
echo.
echo [2/4] Обновление переменной PATH...
set "PATH=%PYTHON_INSTALL_DIR%;%PYTHON_INSTALL_DIR%\Scripts;%PATH%"

:: Проверяем, что Python работает
%PYTHON_EXE% --version
if %errorLevel% neq 0 (
    echo [ERROR] Python не найден после установки.
    pause
    exit /b 1
)

:: Шаг 3: Устанавливаем зависимости из requirements.txt
echo.
echo [3/4] Установка зависимостей из requirements.txt...
if not exist "requirements.txt" (
    echo [WARN] Файл requirements.txt не найден в текущей папке.
    echo Пропускаем установку зависимостей.
) else (
    echo Установка пакетов через pip...
    %PYTHON_EXE% -m pip install --upgrade pip
    %PYTHON_EXE% -m pip install -r requirements.txt
    if %errorLevel% neq 0 (
        echo [ERROR] Ошибка при установке зависимостей.
        pause
        exit /b 1
    )
    echo [OK] Зависимости установлены.
)

:: Шаг 4: Запускаем server.py
echo.
echo [4/4] Запуск server.py...
if not exist "server.py" (
    echo [ERROR] Файл server.py не найден в текущей папке.
    echo Убедитесь, что server.py находится в той же директории, что и этот батник.
    pause
    exit /b 1
)

echo Запуск сервера...
%PYTHON_EXE% server.py

:: Если сервер завершился, показываем сообщение
echo.
echo Сервер остановлен.
pause