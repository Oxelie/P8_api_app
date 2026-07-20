FROM python:3.11-slim

# Dépendances système pour opencv et albumentations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances Python en premier (layer mis en cache si requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code applicatif
COPY api/ ./api/
COPY Procfile .

# Port exposé par gunicorn
ENV PORT=8000
EXPOSE 8000

CMD gunicorn "api.api:app" --bind 0.0.0.0:$PORT --timeout 300 --workers 1
