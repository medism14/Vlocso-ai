from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import random
import os
import surprise
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI()

# Configurer le répertoire de données Surprise
os.environ['SURPRISE_DATA_FOLDER'] = os.environ.get('SURPRISE_DATA_FOLDER', '/home/app/.surprise_data')

# Initialiser les variables globales
annonces_df = None
vehicles_df = None
interactions_df = None
annonces_vehicles_df = None
model_svd = None

class RecommendationRequest(BaseModel):
    excludeIds: Optional[List[int]] = []
    nbAnnonces: Optional[int] = 12

@app.on_event("startup")
async def load_data():
    """
    Charge les données au démarrage de l'application
    """
    global annonces_df, vehicles_df, interactions_df, annonces_vehicles_df, model_svd
    
    try:
        # Charger les données
        annonces_df = pd.read_csv("annonces.csv")
        vehicles_df = pd.read_csv("vehicles.csv")
        interactions_df = pd.read_csv("interactions.csv")
        
        # Fusion des données avec sélection optimisée des colonnes
        annonces_vehicles_df = pd.merge(
            annonces_df,
            vehicles_df[['vehicle_id', 'type', 'mark', 'model', 'category']],
            on='vehicle_id',
            how='left'
        )
        
        # Charger le modèle pré-entraîné
        model_svd = joblib.load('svd_model.pkl')
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        raise e

def get_similar_annonces(annonce, filtered_df, price_range=0.2):
    """
    Trouve les annonces similaires basées sur plusieurs critères
    """
    price_min = float(annonce['price']) * (1 - price_range)
    price_max = float(annonce['price']) * (1 + price_range)
    
    similar = filtered_df[
        ((filtered_df['type'] == annonce['type']) & 
         (filtered_df['price'].between(price_min, price_max))) |
        ((filtered_df['mark'] == annonce['mark']) & 
         (filtered_df['type'] == annonce['type'])) |
        ((filtered_df['model'] == annonce['model']) & 
         (filtered_df['type'] == annonce['type'])) |
        ((filtered_df['category'] == annonce['category']) & 
         (filtered_df['type'] == annonce['type']))
    ]
    
    # Prendre 30% des annonces similaires aléatoirement
    n_select = max(int(len(similar) * 0.3), 1)
    return similar.sample(n=min(n_select, len(similar)))

def get_top_interactions(df, interaction_counts, percentage=0.3, min_count=5):
    """
    Retourne les meilleures annonces basées sur les interactions
    """
    df_with_counts = df.merge(
        interaction_counts,
        on='annonce_id',
        how='left'
    ).fillna({'count': 0})
    
    # Prendre 30% des meilleures annonces
    top_count = max(int(len(df_with_counts) * percentage), min_count)
    top_annonces = df_with_counts.nlargest(top_count, 'count')
    
    # Retourner un échantillon aléatoire des meilleures annonces
    return top_annonces.sample(frac=1)

def get_recommendations(user_id: int, vehicle_type: str = None, exclude_ids: List[int] = None, nb_annonces: int = 12):
    if exclude_ids is None:
        exclude_ids = []
        
    try:
        # Filtrage initial des annonces
        filtered_df = annonces_vehicles_df[~annonces_vehicles_df['annonce_id'].isin(exclude_ids)]
        
        # Exclure les annonces de l'utilisateur
        filtered_df = filtered_df[filtered_df['user_id'] != user_id]
        
        if vehicle_type:
            filtered_df = filtered_df[filtered_df['type'] == vehicle_type]
            
        if len(filtered_df) < nb_annonces:
            return []
            
        # Obtenir les interactions de l'utilisateur
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        interaction_counts = interactions_df.groupby('annonce_id').size().reset_index(name='count')
        
        # Pour les recommandations générales, assurer une diversité des types
        if vehicle_type is None:
            # Cas 1: Aucune interaction - Sélectionner aléatoirement parmi les plus populaires
            if len(user_interactions) == 0:
                top_annonces = get_top_interactions(filtered_df, interaction_counts)
                voitures = top_annonces[top_annonces['type'] == 'Voiture']['annonce_id'].tolist()
                motos = top_annonces[top_annonces['type'] == 'Moto']['annonce_id'].tolist()
                
                # Mélanger les listes
                random.shuffle(voitures)
                random.shuffle(motos)
                
                # Répartir équitablement entre voitures et motos
                voitures_count = nb_annonces // 2
                motos_count = nb_annonces - voitures_count
                
                voitures = voitures[:voitures_count]
                motos = motos[:motos_count]
                
                # Mélanger l'ordre des types
                recommendations = voitures + motos
                random.shuffle(recommendations)
                
                return recommendations
                
            # Cas 2: Peu d'interactions (1-3)
            if len(user_interactions) <= 3:
                last_annonce = filtered_df[filtered_df['annonce_id'] == user_interactions.iloc[-1]['annonce_id']].iloc[0]
                similar_annonces = get_similar_annonces(last_annonce, filtered_df)
                similar_annonces = similar_annonces[~similar_annonces['annonce_id'].isin(user_interactions['annonce_id'])]
                
                if len(similar_annonces) >= nb_annonces:
                    return similar_annonces['annonce_id'].tolist()[:nb_annonces]
                    
                # Si pas assez d'annonces similaires, compléter avec des annonces populaires
                top_annonces = get_top_interactions(filtered_df, interaction_counts)
                remaining = nb_annonces - len(similar_annonces)
                recommendations = similar_annonces['annonce_id'].tolist()
                recommendations.extend(top_annonces['annonce_id'].tolist()[:remaining])
                return recommendations
            
            # Cas 3: Plusieurs interactions - Utiliser SVD
            viewed_annonces = set(user_interactions['annonce_id'])
            available_annonces = set(filtered_df['annonce_id']) - viewed_annonces - set(exclude_ids)
            
            if len(available_annonces) < nb_annonces:
                return []
                
            # Prédictions SVD avec batch processing
            predictions = []
            batch_size = 1000
            annonces_list = list(available_annonces)
            
            for i in range(0, len(annonces_list), batch_size):
                batch = annonces_list[i:i + batch_size]
                batch_predictions = model_svd.test([(user_id, annonce_id, 0) for annonce_id in batch])
                predictions.extend(batch_predictions)
                
            # Sélectionner les 30% meilleures prédictions
            top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:int(len(predictions) * 0.3)]
            
            # Séparer par type et mélanger
            voitures = [int(p.iid) for p in top_predictions if filtered_df[filtered_df['annonce_id'] == int(p.iid)]['type'].iloc[0] == 'Voiture']
            motos = [int(p.iid) for p in top_predictions if filtered_df[filtered_df['annonce_id'] == int(p.iid)]['type'].iloc[0] == 'Moto']
            
            random.shuffle(voitures)
            random.shuffle(motos)
            
            # Répartir équitablement
            voitures_count = nb_annonces // 2
            motos_count = nb_annonces - voitures_count
            
            voitures = voitures[:voitures_count]
            motos = motos[:motos_count]
            
            # Mélanger l'ordre des types
            recommendations = voitures + motos
            random.shuffle(recommendations)
            
            return recommendations
            
        else:
            # Pour les recommandations spécifiques à un type
            return get_recommendations_by_type(user_id, filtered_df, user_interactions, interaction_counts, nb_annonces)
            
    except Exception as e:
        print(f"Erreur lors de la génération des recommandations: {str(e)}")
        return []

def get_recommendations_by_type(user_id, filtered_df, user_interactions, interaction_counts, nb_annonces):
    """
    Fonction auxiliaire pour les recommandations spécifiques à un type
    """
    if len(user_interactions) == 0:
        top_annonces = get_top_interactions(filtered_df, interaction_counts)
        return random.sample(top_annonces['annonce_id'].tolist()[:int(len(top_annonces)*0.3)], nb_annonces)
        
    if len(user_interactions) <= 3:
        last_annonce = filtered_df[filtered_df['annonce_id'] == user_interactions.iloc[-1]['annonce_id']].iloc[0]
        similar_annonces = get_similar_annonces(last_annonce, filtered_df)
        similar_annonces = similar_annonces[~similar_annonces['annonce_id'].isin(user_interactions['annonce_id'])]
        
        if len(similar_annonces) >= nb_annonces:
            return random.sample(similar_annonces['annonce_id'].tolist(), nb_annonces)
            
        top_annonces = get_top_interactions(filtered_df, interaction_counts)
        return random.sample(top_annonces['annonce_id'].tolist()[:int(len(top_annonces)*0.3)], nb_annonces)
        
    viewed_annonces = set(user_interactions['annonce_id'])
    available_annonces = set(filtered_df['annonce_id']) - viewed_annonces
    
    predictions = []
    batch_size = 1000
    annonces_list = list(available_annonces)
    
    for i in range(0, len(annonces_list), batch_size):
        batch = annonces_list[i:i + batch_size]
        batch_predictions = model_svd.test([(user_id, annonce_id, 0) for annonce_id in batch])
        predictions.extend(batch_predictions)
        
    # Sélectionner aléatoirement parmi les 30% meilleures prédictions
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:int(len(predictions) * 0.3)]
    selected_predictions = random.sample(top_predictions, nb_annonces)
    
    return [int(pred.iid) for pred in selected_predictions]

@app.get("/")
async def main():
    return {
        "Server is running up"
    }

@app.post("/recommend/general/{user_id}")
async def get_general_recommendations(user_id: int, request: RecommendationRequest):
    """
    Endpoint pour obtenir des recommandations générales.
    """
    try:
        recommendations = get_recommendations(user_id, exclude_ids=request.excludeIds, nb_annonces=request.nbAnnonces)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/voitures/{user_id}")
async def get_car_recommendations(user_id: int, request: RecommendationRequest):
    """
    Endpoint pour obtenir des recommandations de voitures.
    """
    try:
        recommendations = get_recommendations(user_id, vehicle_type='Voiture', exclude_ids=request.excludeIds, nb_annonces=request.nbAnnonces)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/motos/{user_id}")
async def get_moto_recommendations(user_id: int, request: RecommendationRequest):
    """
    Endpoint pour obtenir des recommandations de motos.
    """
    try:
        recommendations = get_recommendations(user_id, vehicle_type='Moto', exclude_ids=request.excludeIds, nb_annonces=request.nbAnnonces)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)