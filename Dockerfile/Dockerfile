# IMAGE DE BASE - Python 3.11 léger

FROM python:3.11-slim

# VARIABLES D'ENVIRONNEMENT

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# RÉPERTOIRE DE TRAVAIL

WORKDIR /app

# INSTALLATION DES DÉPENDANCES SYSTÈME

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# INSTALLATION DES DÉPENDANCES PYTHON

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# COPIE DU CODE SOURCE

COPY . .

# UTILISATEUR NON-ROOT

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# EXPOSITION ET DÉMARRAGE

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py"]
