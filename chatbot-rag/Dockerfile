FROM python:3.11.7-slim-bookworm

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    sqlite3 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 80

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "80"]