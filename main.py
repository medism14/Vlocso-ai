from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import random
import os
import surprise

app = FastAPI()

# Configurer le répertoire de données Surprise
os.environ['SURPRISE_DATA_FOLDER'] = os.environ.get('SURPRISE_DATA_FOLDER', '/home/app/.surprise_data')

# Initialiser les variables globales
annonces_df = None
vehicles_df = None
interactions_df = None
annonces_vehicles_df = None
model_svd = None

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

def get_recommendations(user_id: int, vehicle_type: str = None, n: int = 12) -> list:
    """
    Génère des recommandations pour un utilisateur donné.
    """
    try:
        # Filtrer par type de véhicule si spécifié
        if vehicle_type:
            filtered_df = annonces_vehicles_df[annonces_vehicles_df['type'] == vehicle_type]
        else:
            filtered_df = annonces_vehicles_df

        # Vérifier les interactions de l'utilisateur
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        interaction_count = len(user_interactions)
        
        # Cas 1: Aucune interaction
        if interaction_count == 0:
            # Retourner les annonces les plus populaires
            popular_annonces = filtered_df.sort_values('price', ascending=False)
            return popular_annonces.head(n)['annonce_id'].tolist()
            
        # Cas 2: Très peu d'interactions (1-3)
        if interaction_count <= 3:
            # Obtenir la dernière interaction
            last_interactions = user_interactions.sort_values('interaction_date', ascending=False)
            last_annonce_id = last_interactions.iloc[0]['annonce_id']
            
            # Trouver l'annonce correspondante
            last_annonce = filtered_df[filtered_df['annonce_id'] == last_annonce_id]
            if not last_annonce.empty:
                annonce = last_annonce.iloc[0]
                
                # Construire des recommandations basées sur les caractéristiques similaires
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
                
                # Exclure les annonces déjà vues
                viewed_annonces = user_interactions['annonce_id'].unique()
                similar_annonces = similar_annonces[~similar_annonces['annonce_id'].isin(viewed_annonces)]
                
                if len(similar_annonces) >= n:
                    return similar_annonces.sample(n)['annonce_id'].tolist()
                else:
                    # Compléter avec des annonces populaires si pas assez de similaires
                    remaining_count = n - len(similar_annonces)
                    popular_annonces = filtered_df[~filtered_df['annonce_id'].isin(similar_annonces['annonce_id'])]
                    popular_annonces = popular_annonces.sort_values('price', ascending=False)
                    
                    recommendations = similar_annonces['annonce_id'].tolist()
                    recommendations.extend(popular_annonces.head(remaining_count)['annonce_id'].tolist())
                    return recommendations
            
            # Si pas d'annonce trouvée, retourner les populaires
            return filtered_df.sort_values('price', ascending=False).head(n)['annonce_id'].tolist()
        
        # Cas 3: Plusieurs interactions - utiliser le modèle SVD
        user_annonces = user_interactions['annonce_id'].unique()
        all_annonces = filtered_df['annonce_id'].unique()
        annonces_to_predict = list(set(all_annonces) - set(user_annonces))
        
        if not annonces_to_predict:
            return filtered_df.sort_values('price', ascending=False).head(n)['annonce_id'].tolist()
        
        # Prédictions avec le modèle
        user_annonce_pairs = [(user_id, annonce_id, 0) for annonce_id in annonces_to_predict]
        predictions = model_svd.test(user_annonce_pairs)
        sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
        
        # Sélection aléatoire parmi les meilleures prédictions
        top_predictions = sorted_predictions[:int(len(sorted_predictions) * 0.3)]
        selected_predictions = random.sample(top_predictions, min(n, len(top_predictions)))
        return [int(pred.iid) for pred in selected_predictions]
        
    except Exception as e:
        print(f"Erreur lors de la génération des recommandations: {str(e)}")
        return []

@app.get("/")
async def main():
    return {
        "Server is running up"
    }

@app.post("/recommend/general/{user_id}")
async def get_general_recommendations(user_id: int):
    """
    Endpoint pour obtenir des recommandations générales.
    """
    try:
        recommendations = get_recommendations(user_id)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/cars/{user_id}")
async def get_car_recommendations(user_id: int):
    """
    Endpoint pour obtenir des recommandations de voitures.
    """
    try:
        recommendations = get_recommendations(user_id, vehicle_type='Voiture')
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/motos/{user_id}")
async def get_moto_recommendations(user_id: int):
    """
    Endpoint pour obtenir des recommandations de motos.
    """
    try:
        recommendations = get_recommendations(user_id, vehicle_type='Moto')
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)