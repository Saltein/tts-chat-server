FROM python:3.12-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install --no-cache-dir gunicorn flask flask-cors soundfile requests

COPY . .

EXPOSE 5001

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "--timeout", "120", "--max-requests", "50", "--max-requests-jitter", "10", "server:app"]
