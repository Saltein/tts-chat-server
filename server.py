from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
import uuid
import os
import logging
import soundfile as sf
import requests
import io
import gc
import threading
from ws_lobby import main as ws_main
import asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = False

# -----------------------------
# Настройка и загрузка Silero TTS
# -----------------------------
logging.info("Загрузка модели Silero TTS...")

device = torch.device("cpu")
TEMP_DIR = "tts_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Скачиваем модель, если её нет
model_path = os.path.join(TEMP_DIR, "model.pt")
if not os.path.isfile(model_path):
    logging.info("Скачивание модели...")
    url = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
    response = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    logging.info("Модель скачана")

# Загружаем модель официальным способом
model = None
try:
    logging.info("Загрузка модели официальным способом...")
    model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
    model.to(device)
    print('-------------------------SPEAKERS-------------------------', model.speakers, '----------------------------------------------------------', sep='\n')
    logging.info("Модель успешно загружена")
except Exception as e:
    logging.error(f"Ошибка загрузки модели: {e}")

# Параметры и доступные голоса
sample_rate = 24000
speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
DEFAULT_SPEAKER = 'aidar'


def text_to_speech(text, speaker=None):
    if speaker is None or speaker not in speakers:
        speaker = DEFAULT_SPEAKER

    if model is None:
        raise Exception("Модель не загружена")

    logging.info(f"Генерация речи: '{text[:30]}...' с голосом '{speaker}'")

    try:
        with torch.no_grad():
            audio = model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=sample_rate
            )

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)

        # Освобождаем память
        del audio
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return buffer

    except Exception as e:
        logging.error(f"Ошибка генерации речи: {e}")
        raise


# -----------------------------
# Эндпоинты
# -----------------------------
@app.route("/api/speak", methods=["POST"])
def speak():
    data = request.get_json()
    if not data or "text" not in data:
        logging.warning("Некорректный запрос: 'text' отсутствует")
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    speaker = data.get("speaker", DEFAULT_SPEAKER)

    if speaker not in speakers:
        logging.warning(f"Запрошенный голос '{speaker}' не найден. Используется голос по умолчанию: {DEFAULT_SPEAKER}")
        speaker = DEFAULT_SPEAKER

    try:
        buffer = text_to_speech(text, speaker)

        response = send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"speech_{speaker}_{uuid.uuid4().hex[:8]}.wav"
        )

        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response

    except Exception as e:
        logging.error(f"Ошибка генерации речи: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    status = "ok" if model is not None else "error"
    return jsonify({
        "status": status,
        "message": "TTS server is running" if model is not None else "Model not loaded"
    })


@app.route("/api/speakers", methods=["GET"])
def get_speakers():
    return jsonify({"speakers": speakers})


# -----------------------------
# Запуск сервера
# -----------------------------
if __name__ == "__main__":
    if model is None:
        logging.error("Не удалось загрузить модель TTS! Сервер не может работать.")
    else:
        # Запускаем WebSocket сервер в отдельном потоке
        def start_ws_server():
            asyncio.run(ws_main())

        threading.Thread(target=start_ws_server, daemon=True).start()

        logging.info("Запуск Flask-сервера на http://0.0.0.0:5001")
        app.run(host="0.0.0.0", port=5001)
