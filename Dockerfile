FROM python:3.12-slim

WORKDIR /app
RUN pip install torch flask flask-cors soundfile cors

COPY . .

EXPOSE 5001
CMD ["python", "server.py"]