import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Chargement
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze()

# Chargement des meilleurs paramètres
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Entraînement
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde du modèle
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé !")
