# Image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers nécessaires
COPY main.py .
COPY svd_model.pkl .
COPY annonces.csv .
COPY vehicles.csv .
COPY interactions.csv .

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 