from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
import uuid
import os
import logging
import soundfile as sf
import requests
import gc
import threading
import time
import tempfile
import asyncio
import websockets
import json
from ws_lobby import main as ws_main
from enum import Enum
import shutil
import sys
import subprocess

# Проверка и установка необходимых зависимостей
def check_and_install_dependencies():
    """Проверка и установка необходимых зависимостей"""
    required_packages = {
        'scipy': 'scipy',
        'numpy': 'numpy',
        'torchaudio': 'torchaudio'
    }
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            logging.info(f"✓ {package} found")
        except ImportError:
            logging.warning(f"{package} not found, installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logging.info(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to install {package}: {e}")
                raise

# Устанавливаем зависимости перед импортом остальных модулей
try:
    check_and_install_dependencies()
except Exception as e:
    logging.error(f"Dependency installation failed: {e}")
    sys.exit(1)

# Импортируем после проверки зависимостей
import scipy.signal
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = False

# -----------------------------
# Типы ошибок
# -----------------------------
class ErrorType(Enum):
    STARTUP = "startup_error"
    GENERATION = "generation_error"
    NETWORK = "network_error"
    MODEL = "model_error"

class StartupError(Exception):
    """Исключение для ошибок запуска сервера"""
    pass

class GenerationError(Exception):
    """Исключение для ошибок генерации речи"""
    pass

# -----------------------------
# Настройка WebSocket клиента
# -----------------------------
WS_SERVER_URL = "ws://127.0.0.1:3036"
websocket_connection = None

async def send_ws_message(message_type, message_data):
    """Отправка сообщения на WebSocket сервер"""
    global websocket_connection
    try:
        if websocket_connection is None:
            websocket_connection = await websockets.connect(WS_SERVER_URL)
        
        message = {
            "type": message_type,
            "data": message_data,
            "timestamp": time.time()
        }
        
        await websocket_connection.send(json.dumps(message))
        logging.info(f"WebSocket message sent: {message_type}")
        
    except Exception as e:
        logging.error(f"Failed to send WebSocket message: {e}")
        websocket_connection = None

def send_status_message(message_type, status, details=None):
    """Утилита для отправки статусных сообщений"""
    message_data = {
        "status": status,
        "service": "TTS Server",
        "details": details or {}
    }
    
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_ws_message(message_type, message_data))
        loop.close()
    
    threading.Thread(target=run_async, daemon=True).start()

def send_error_message(error_type: ErrorType, error_message: str, error_details=None, is_fatal=False):
    """Отправка структурированного сообщения об ошибке"""
    error_data = {
        "error_type": error_type.value,
        "error_category": "startup" if error_type in [ErrorType.STARTUP, ErrorType.MODEL] else "generation",
        "message": error_message,
        "is_fatal": is_fatal,
        "details": error_details or {}
    }
    
    send_status_message("tts-error", "error", error_data)
    logging.error(f"[{error_type.value}] {error_message}")

# -----------------------------
# Пути к файлам (исправлено для production)
# -----------------------------
def get_base_dir():
    """Возвращает базовую директорию для ресурсов (модели)"""
    if getattr(sys, 'frozen', False):
        # Запущено как exe – ресурсы лежат в _MEIPASS
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

# Папка для модели (только чтение)
MODEL_DIR = get_base_dir()
MODEL_PATH = os.path.join(MODEL_DIR, "model5.pt")

# Папка для временных файлов (системная temp папка)
TEMP_BASE = tempfile.gettempdir()
APP_TEMP_DIR = os.path.join(TEMP_BASE, "tts_electron")
SOUNDS_DIR = os.path.join(APP_TEMP_DIR, "sounds")

# Создаём директории для временных файлов
os.makedirs(SOUNDS_DIR, exist_ok=True)

logging.info(f"Model path: {MODEL_PATH}")
logging.info(f"Temporary sounds directory: {SOUNDS_DIR}")

# -----------------------------
# Альтернативный метод загрузки Silero TTS
# -----------------------------
def load_silero_model_alternative():
    """Альтернативный метод загрузки модели через silero library"""
    try:
        # Попробуем использовать официальный метод через silero
        import torchaudio
        from silero import silero_tts
        
        logging.info("Attempting to load model via silero_tts...")
        model = silero_tts(language='ru', speaker='v3_ru')
        return model, 24000, model.speakers
        
    except ImportError:
        logging.warning("silero package not found, using fallback method")
        return None, None, None
    except Exception as e:
        logging.warning(f"Alternative loading failed: {e}")
        return None, None, None

# -----------------------------
# Настройка и загрузка Silero TTS
# -----------------------------
logging.info("Загрузка модели Silero TTS...")
send_status_message("tts-start", "loading", {"message": "TTS model loading started"})

device = torch.device("cpu")
model = None
sample_rate = 24000
speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
DEFAULT_SPEAKER = 'aidar'
TEMP_FILE_LIFETIME = 60  # Увеличил до 60 секунд

def get_random_speaker():
    """Возвращает случайный голос из доступных"""
    import random
    return random.choice(speakers)

# Пытаемся загрузить модель
try:
    # Проверяем наличие модели
    if not os.path.isfile(MODEL_PATH):
        logging.warning(f"Model not found at {MODEL_PATH}, attempting to download...")
        send_status_message("tts-download", "downloading", {"message": "Downloading TTS model"})
        
        try:
            url = "https://models.silero.ai/models/tts/ru/v5_ru.pt"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # В production модель скачать не сможем (read-only), но попробуем
            # Если не получится, выбросим ошибку
            try:
                with open(MODEL_PATH, 'wb') as f:
                    f.write(response.content)
                logging.info("Model downloaded successfully")
                send_status_message("tts-download", "success", {"message": "TTS model downloaded successfully"})
            except (PermissionError, OSError):
                error_msg = f"Cannot write model to {MODEL_PATH} (read-only filesystem)"
                logging.error(error_msg)
                raise StartupError(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error while downloading model: {e}"
            logging.error(error_msg)
            send_error_message(ErrorType.NETWORK, error_msg, {"url": url, "error": str(e)}, is_fatal=True)
            raise StartupError(error_msg)
        except Exception as e:
            error_msg = f"Failed to download model: {e}"
            logging.error(error_msg)
            send_error_message(ErrorType.STARTUP, error_msg, {"stage": "download", "error": str(e)}, is_fatal=True)
            raise StartupError(error_msg)

    # Загружаем модель с обработкой зависимостей
    logging.info("Loading model...")
    send_status_message("tts-load", "loading", {"message": "Loading TTS model"})
    
    # Устанавливаем необходимые глобальные переменные для torch.package
    sys.modules['scipy.signal'] = scipy.signal
    sys.modules['numpy'] = np
    
    # Загружаем модель
    try:
        model = torch.package.PackageImporter(MODEL_PATH).load_pickle("tts_models", "model")
        model.to(device)
        logging.info("Model successfully loaded via torch.package")
    except Exception as e:
        logging.warning(f"torch.package loading failed: {e}, trying alternative method...")
        # Пробуем альтернативный метод
        model, sample_rate, speakers = load_silero_model_alternative()
        if model is None:
            raise
    
    logging.info("Model successfully loaded")
    
    send_status_message("tts-ready", "success", {
        "message": "TTS model loaded successfully",
        "speakers": speakers,
        "sample_rate": sample_rate,
        "device": str(device),
        "temp_directory": SOUNDS_DIR
    })
    
except Exception as e:
    error_msg = f"Model loading error: {e}"
    logging.error(error_msg)
    send_error_message(ErrorType.MODEL, error_msg, {
        "stage": "loading",
        "model_path": MODEL_PATH,
        "error": str(e)
    }, is_fatal=True)
    raise StartupError(error_msg)

# -----------------------------
# Функции генерации TTS
# -----------------------------
def text_to_speech_file(text, speaker=None):
    """Генерирует речь из текста и сохраняет во временный файл"""
    
    if speaker == 'random':
        speaker = get_random_speaker()
        logging.info(f"Random speaker selected: {speaker}")
        
    if speaker not in speakers:
        speaker = DEFAULT_SPEAKER

    # Создаём временный файл в SOUNDS_DIR
    try:
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=".wav", 
            dir=SOUNDS_DIR
        )
        temp_file_path = temp_file.name
        temp_file.close()
    except Exception as e:
        error_msg = f"Failed to create temporary file in {SOUNDS_DIR}: {e}"
        logging.error(error_msg)
        raise GenerationError(error_msg)

    logging.info(f"Generating speech: '{text[:30]}...' with voice '{speaker}'")
    logging.info(f"Output file: {temp_file_path}")
    
    send_status_message("tts-generating", "processing", {
        "message": f"Generating speech for text: {text[:50]}...",
        "speaker": speaker,
        "text_length": len(text),
        "output_file": os.path.basename(temp_file_path)
    })
    
    try:
        # Проверяем модель
        if model is None:
            raise GenerationError("TTS model is not loaded")
        
        # Генерируем аудио
        audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
        
        # Проверяем результат
        if audio is None or len(audio) == 0:
            raise GenerationError("Generated audio is empty")
        
        # Сохраняем файл
        sf.write(temp_file_path, audio, sample_rate)
        
        # Проверяем, что файл действительно создан
        if not os.path.exists(temp_file_path):
            raise GenerationError(f"File was not created: {temp_file_path}")
        
        file_size = os.path.getsize(temp_file_path)
        logging.info(f"File created successfully, size: {file_size} bytes")

        # Очищаем память
        del audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Планируем удаление файла
        threading.Thread(target=delete_file_later, args=(temp_file_path, TEMP_FILE_LIFETIME), daemon=True).start()
        
        send_status_message("tts-generated", "success", {
            "message": "Speech generated successfully",
            "speaker": speaker,
            "file": os.path.basename(temp_file_path),
            "file_size": file_size
        })

        return temp_file_path
        
    except GenerationError as e:
        error_msg = str(e)
        logging.error(f"Generation error: {error_msg}")
        send_error_message(ErrorType.GENERATION, error_msg, {
            "speaker": speaker,
            "text_preview": text[:100],
            "stage": "generation"
        })
        raise
        
    except torch.cuda.OutOfMemoryError as e:
        error_msg = "GPU out of memory during generation"
        logging.error(f"GPU OOM: {e}")
        send_error_message(ErrorType.GENERATION, error_msg, {
            "speaker": speaker,
            "error": str(e),
            "stage": "memory_allocation"
        })
        raise GenerationError(error_msg)
        
    except RuntimeError as e:
        error_msg = f"Runtime error during TTS generation: {e}"
        logging.error(error_msg)
        send_error_message(ErrorType.GENERATION, error_msg, {
            "speaker": speaker,
            "text_preview": text[:100],
            "error_type": "runtime",
            "stage": "model_inference"
        })
        raise GenerationError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error during generation: {e}"
        logging.error(error_msg)
        send_error_message(ErrorType.GENERATION, error_msg, {
            "speaker": speaker,
            "text_preview": text[:100],
            "error_type": type(e).__name__,
            "stage": "unknown"
        })
        raise GenerationError(error_msg)

def delete_file_later(path, delay):
    """Удаляет файл через заданную задержку"""
    time.sleep(delay)
    try:
        if os.path.isfile(path):
            os.remove(path)
            logging.info(f"Temporary file deleted: {path}")
    except Exception as e:
        logging.error(f"Error deleting temporary file {path}: {e}")

# -----------------------------
# Эндпоинты
# -----------------------------
@app.route("/api/speak", methods=["POST"])
def speak():
    """Генерация речи из текста"""
    try:
        data = request.get_json()
        
        # Валидация входных данных
        if not data:
            raise GenerationError("Empty request body")
        
        if "text" not in data:
            raise GenerationError("Missing 'text' in request")
        
        text = data["text"]
        if not text or not isinstance(text, str):
            raise GenerationError("Text must be a non-empty string")
        
        if len(text) > 1000:
            raise GenerationError(f"Text too long: {len(text)} characters (max 1000)")
        
        speaker = data.get("speaker", DEFAULT_SPEAKER)
        if speaker not in speakers:
            logging.warning(f"Requested speaker '{speaker}' not found. Using default: {DEFAULT_SPEAKER}")
            speaker = DEFAULT_SPEAKER

        # Генерация речи
        file_path = text_to_speech_file(text, speaker)

        response = send_file(
            file_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"speech_{speaker}_{uuid.uuid4().hex[:8]}.wav"
        )

        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response

    except GenerationError as e:
        # Ошибки генерации - возвращаем 400/422
        return jsonify({
            "error": {
                "type": "generation_error",
                "message": str(e),
                "category": "generation"
            }
        }), 422
        
    except Exception as e:
        # Неожиданные ошибки
        logging.error(f"Unexpected error in speak endpoint: {e}")
        return jsonify({
            "error": {
                "type": "internal_error",
                "message": "Internal server error occurred",
                "category": "unknown"
            }
        }), 500

@app.route("/api/health", methods=["GET"])
def health():
    """Проверка здоровья сервера с детализацией"""
    health_status = {
        "status": "ok" if model is not None else "error",
        "message": "TTS server is running" if model is not None else "Model not loaded",
        "checks": {
            "model_loaded": model is not None,
            "temp_dir_writable": os.access(SOUNDS_DIR, os.W_OK),
            "speakers_available": len(speakers) if model is not None else 0,
            "temp_directory": SOUNDS_DIR
        }
    }
    
    if model is None:
        health_status["error_type"] = "startup_error"
        health_status["error_category"] = "startup"
    
    return jsonify(health_status)

@app.route("/api/speakers", methods=["GET"])
def get_speakers():
    """Возвращает список доступных голосов"""
    return jsonify({"speakers": speakers})

@app.route("/api/status", methods=["GET"])
def get_status():
    """Детальный статус сервера"""
    return jsonify({
        "server": {
            "status": "running",
            "startup_completed": model is not None,
            "uptime": time.time() - app.config.get("START_TIME", time.time())
        },
        "model": {
            "loaded": model is not None,
            "speakers": speakers if model is not None else [],
            "sample_rate": sample_rate,
            "device": str(device)
        },
        "temp_directory": {
            "path": SOUNDS_DIR,
            "writable": os.access(SOUNDS_DIR, os.W_OK),
            "exists": os.path.exists(SOUNDS_DIR)
        },
        "errors": {
            "has_startup_errors": model is None,
            "has_generation_errors": False
        }
    })

# -----------------------------
# Очистка при завершении
# -----------------------------
def cleanup_temp_files():
    """Очищает временные файлы при завершении сервера"""
    try:
        if os.path.exists(SOUNDS_DIR):
            files = os.listdir(SOUNDS_DIR)
            for file in files:
                file_path = os.path.join(SOUNDS_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logging.error(f"Error cleaning up {file_path}: {e}")
            logging.info(f"Cleaned up {len(files)} temporary files")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

import atexit
atexit.register(cleanup_temp_files)

# -----------------------------
# Запуск сервера
# -----------------------------
if __name__ == "__main__":
    app.config["START_TIME"] = time.time()
    
    # Флаг, ожидающий готовности WebSocket сервера
    websocket_ready = False
    max_retries = 30
    retry_count = 0
    
    # Проверяем доступность WebSocket сервера перед запуском
    while not websocket_ready and retry_count < max_retries:
        try:
            # Пытаемся подключиться к WebSocket серверу
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 3036))
            sock.close()
            
            if result == 0:
                websocket_ready = True
                logging.info("WebSocket server is ready")
                
                # Отправляем сообщение о готовности TTS сервера
                async def send_ready():
                    await send_ws_message("tts-server-ready", {
                        "status": "starting",
                        "message": "TTS Server is starting, WebSocket connection established"
                    })
                
                asyncio.run(send_ready())
                break
            else:
                retry_count += 1
                logging.info(f"Waiting for WebSocket server... ({retry_count}/{max_retries})")
                time.sleep(1)
                
        except Exception as e:
            retry_count += 1
            logging.warning(f"WebSocket server not ready yet: {e} ({retry_count}/{max_retries})")
            time.sleep(1)
    
    if not websocket_ready:
        logging.warning("WebSocket server not available after maximum retries. Continuing without WebSocket...")
    
    # Сообщение о старте
    send_status_message("tts-server-start", "starting", {
        "message": "TTS Server is starting",
        "port": 5001,
        "host": "0.0.0.0",
        "temp_directory": SOUNDS_DIR
    })
    
    if model is None:
        error_msg = "Failed to load TTS model. Server cannot start."
        logging.error(error_msg)
        send_error_message(ErrorType.STARTUP, error_msg, {"fatal": True}, is_fatal=True)
        logging.error("Server startup aborted due to fatal error")
        sys.exit(1)
    
    try:
        # Запускаем WebSocket клиент для связи с основным сервером
        def start_websocket_client():
            """Запускает WebSocket клиент для связи с основным приложением"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def maintain_connection():
                global websocket_connection
                while True:
                    try:
                        if websocket_connection is None:
                            websocket_connection = await websockets.connect(WS_SERVER_URL)
                            logging.info("WebSocket client connected to main server")
                            
                            # Отправляем сообщение о готовности
                            await send_ws_message("tts-server-ready", {
                                "status": "ready",
                                "message": "TTS Server is fully operational",
                                "temp_directory": SOUNDS_DIR
                            })
                        
                        await asyncio.sleep(30)  # Keep connection alive
                        
                    except Exception as e:
                        logging.error(f"WebSocket client error: {e}")
                        websocket_connection = None
                        await asyncio.sleep(5)  # Retry after 5 seconds
            
            try:
                loop.run_until_complete(maintain_connection())
            except Exception as e:
                logging.error(f"WebSocket client thread error: {e}")
        
        # Запускаем WebSocket клиент в отдельном потоке (если websocket_ready)
        if websocket_ready:
            ws_client_thread = threading.Thread(target=start_websocket_client, daemon=True)
            ws_client_thread.start()
            logging.info("WebSocket client thread started")
        else:
            logging.info("WebSocket client disabled (server not available)")
        
        logging.info(f"Starting Flask server on http://0.0.0.0:5001")
        logging.info(f"Using temporary directory: {SOUNDS_DIR}")
        
        # Сообщение об успешном запуске
        send_status_message("tts-server-ready", "ready", {
            "message": "TTS Server is ready to accept requests",
            "port": 5001,
            "host": "0.0.0.0",
            "endpoints": [
                "/api/speak",
                "/api/health", 
                "/api/speakers",
                "/api/status"
            ],
            "speakers_available": speakers,
            "model_loaded": True,
            "temp_directory": SOUNDS_DIR
        })
        
        # Запуск Flask
        app.run(host="0.0.0.0", port=5001, use_reloader=False, threaded=True)
        
    except Exception as e:
        error_msg = f"Failed to start Flask server: {e}"
        logging.error(error_msg)
        send_error_message(ErrorType.STARTUP, error_msg, {
            "stage": "flask_startup",
            "error": str(e)
        }, is_fatal=True)
        sys.exit(1)