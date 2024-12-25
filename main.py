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
        
        # Fusion des données
        annonces_vehicles_df = pd.merge(
            annonces_df, 
            vehicles_df[['vehicle_id', 'type']], 
            on='vehicle_id', 
            how='left'
        )
        
        # Charger le modèle pré-entraîné
        model_svd = joblib.load('svd_model.pkl')
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        raise e

def get_recommendations(user_id: int, vehicle_type: str = None, exclude_ids: List[int] = None):
    if exclude_ids is None:
        exclude_ids = []
    
    try:
        if vehicle_type:
            filtered_df = annonces_vehicles_df[annonces_vehicles_df['type'] == vehicle_type]
        else:
            filtered_df = annonces_vehicles_df

        filtered_df = filtered_df[~filtered_df['annonce_id'].isin(exclude_ids)]
        
        # Si moins de 12 annonces disponibles après filtrage, retourner une liste vide
        if len(filtered_df) < 12:
            return []
            
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        interaction_count = len(user_interactions)
        
        # Cas 1: Aucune interaction - Prendre les 30% des annonces avec le plus d'interactions et en sélectionner 12 au hasard
        if len(user_interactions) == 0:
            # Compter le nombre d'interactions par annonce
            interaction_counts = interactions_df.groupby('annonce_id').size().reset_index(name='count')
            
            # Joindre avec filtered_df pour avoir les annonces filtrées avec leur nombre d'interactions
            filtered_with_counts = filtered_df.merge(
                interaction_counts,
                on='annonce_id',
                how='left'
            ).fillna({'count': 0})
            
            # Prendre les 30% des annonces avec le plus d'interactions
            top_count = int(len(filtered_with_counts) * 0.3)
            top_annonces = filtered_with_counts.nlargest(top_count, 'count')
            
            # Pour chaque cas, vérifier s'il y a assez d'annonces
            if len(top_annonces) < 12:
                return []
            
            return random.sample(top_annonces['annonce_id'].tolist(), min(12, len(top_annonces)))
            
        # Cas 2: Très peu d'interactions (1-3)
        if interaction_count <= 3:
            # Compter le nombre d'interactions par annonce
            interaction_counts = interactions_df.groupby('annonce_id').size().reset_index(name='count')
            
            # Récupérer la dernière annonce consultée
            last_interactions = user_interactions.sort_values('interaction_date', ascending=False)
            last_annonce_id = last_interactions.iloc[0]['annonce_id']
            last_annonce = filtered_df[filtered_df['annonce_id'] == last_annonce_id]
            
            if not last_annonce.empty:
                annonce = last_annonce.iloc[0]
                
                # Filtrer les annonces similaires
                similar_annonces = filtered_df[
                    (filtered_df['type'] == annonce['type']) |
                    (filtered_df['mark'] == annonce['mark']) |
                    (filtered_df['model'] == annonce['model']) |
                    (filtered_df['category'] == annonce['category']) |
                    (
                        (filtered_df['price'].between(
                            float(annonce['price']) * 0.8, 
                            float(annonce['price']) * 1.2
                        )) &
                        (filtered_df['type'] == annonce['type'])
                    )
                ]
                
                # Joindre avec le nombre d'interactions
                similar_with_counts = similar_annonces.merge(
                    interaction_counts,
                    on='annonce_id',
                    how='left'
                ).fillna({'count': 0})
                
                # Exclure les annonces déjà vues
                viewed_annonces = user_interactions['annonce_id'].unique()
                similar_with_counts = similar_with_counts[~similar_with_counts['annonce_id'].isin(viewed_annonces)]
                
                # Prendre les 30% des annonces similaires avec le plus d'interactions
                top_count = int(len(similar_with_counts) * 0.3)
                top_similar = similar_with_counts.nlargest(top_count, 'count')
                
                if len(top_similar) >= 12:
                    return random.sample(top_similar['annonce_id'].tolist(), 12)
                else:
                    return top_similar['annonce_id'].tolist()
            
            # Si pas d'annonce similaire trouvée, utiliser la même logique que le cas 1
            filtered_with_counts = filtered_df.merge(
                interaction_counts,
                on='annonce_id',
                how='left'
            ).fillna({'count': 0})
            
            top_count = int(len(filtered_with_counts) * 0.3)
            top_annonces = filtered_with_counts.nlargest(top_count, 'count')
            return random.sample(top_annonces['annonce_id'].tolist(), min(12, len(top_annonces)))
        
        # Cas 3: Plusieurs interactions - utiliser le modèle SVD
        user_annonces = user_interactions['annonce_id'].unique()
        all_annonces = filtered_df['annonce_id'].unique()
        annonces_to_predict = list(set(all_annonces) - set(user_annonces) - set(exclude_ids))
        
        if not annonces_to_predict:
            top_count = int(len(filtered_df) * 0.3)
            top_annonces = filtered_df.merge(
                interaction_counts,
                on='annonce_id',
                how='left'
            ).fillna({'count': 0}).nlargest(top_count, 'count')
            return random.sample(top_annonces['annonce_id'].tolist(), min(12, len(top_annonces)))
        
        # Pour le cas des prédictions SVD
        if len(annonces_to_predict) < 12:
            return []
            
        # Prédictions avec le modèle
        predictions = model_svd.test([(user_id, annonce_id, 0) for annonce_id in annonces_to_predict])
        sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
        
        # Prendre les 30% meilleures prédictions
        top_count = int(len(sorted_predictions) * 0.3)
        top_predictions = sorted_predictions[:top_count]
        
        # Sélectionner 12 prédictions au hasard parmi les meilleures
        selected_predictions = random.sample(top_predictions, min(12, len(top_predictions)))
        recommendations = [int(pred.iid) for pred in selected_predictions]
        
        return recommendations[:12]
        
    except Exception as e:
        print(f"Erreur lors de la génération des recommandations: {str(e)}")
        return []

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
        recommendations = get_recommendations(user_id, exclude_ids=request.excludeIds)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/cars/{user_id}")
async def get_car_recommendations(user_id: int, request: RecommendationRequest):
    """
    Endpoint pour obtenir des recommandations de voitures.
    """
    try:
        recommendations = get_recommendations(user_id, vehicle_type='Voiture', exclude_ids=request.excludeIds)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/motos/{user_id}")
async def get_moto_recommendations(user_id: int, request: RecommendationRequest):
    """
    Endpoint pour obtenir des recommandations de motos.
    """
    try:
        recommendations = get_recommendations(user_id, vehicle_type='Moto', exclude_ids=request.excludeIds)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)