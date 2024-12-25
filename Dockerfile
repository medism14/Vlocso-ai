# Image de base Python
FROM python:3.9-slim

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root avec un répertoire home spécifique
RUN useradd -m -d /home/app app

# Définir le répertoire de travail
WORKDIR /app

# Mettre à jour pip
RUN pip install --no-cache-dir --upgrade pip

# Copier les fichiers de requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Créer le répertoire pour les données Surprise et donner les permissions
RUN mkdir -p /home/app/.surprise_data && \
    chown -R app:app /home/app/.surprise_data

# Copier les fichiers nécessaires
COPY main.py .
COPY svd_model.pkl .
COPY annonces.csv .
COPY vehicles.csv .
COPY interactions.csv .

# Changer le propriétaire des fichiers
RUN chown -R app:app /app

# Définir la variable d'environnement pour Surprise
ENV SURPRISE_DATA_FOLDER=/home/app/.surprise_data

# Utiliser l'utilisateur non-root
USER app

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Variables d'environnement pour la production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"] 