FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# XGBoost nécessite OpenMP (libgomp) sous Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances d'abord : cette couche est mise en cache tant que les versions ne changent pas
COPY requirements-api.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements-api.txt

# Code de l'API, modèle et métadonnées
COPY api ./api
COPY models/model.joblib ./models/model.joblib
COPY reports/metrics.json ./reports/metrics.json

# Utilisateur non root + dossier de logs
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]