import pandas as pd
from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

df =  pd.read_csv('df_c.csv')

# Chargement du modèle
loaded_model = load('reg_model.joblib')



# Création d'une nouvelle instance fastAPI
app = FastAPI()

# Définir un objet (une classe) pour réaliser des requêtes
# dot notation (.)
class request_body(BaseModel):
    lemmatize_joined : str

# Definition du chemin du point de terminaison (API)
@app.post("/predict") # 



# Définition de la fonction de prédiction
def predict(data : request_body):
    # Nouvelles données sur lesquelles on fait la prédiction
    new_data = [[
        data.lemmatize_joined       
    ]]

    # Prédiction uvicorn main:app --reload
    class_idx = loaded_model.predict(new_data)[0]

    # Je retourne si le twitt est positif ou negatif
    return {'target' : df.target[class_idx]}
    
if __name__ == '__main__':
    uvicorn.run('myapp:app', host='0.0.0.0', port=8000)
    uvicorn.run('main:app', host='0.0.0.0', port=8000)

