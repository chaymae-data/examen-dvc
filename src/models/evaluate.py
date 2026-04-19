import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
import os

# Chargement
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze()

# Chargement du modèle
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Prédictions
y_pred = model.predict(X_test)

# Métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE : {mse:.4f}")
print(f"R2  : {r2:.4f}")
print(f"MAE : {mae:.4f}")

# Sauvegarde des métriques
os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump({"mse": mse, "r2": r2, "mae": mae}, f, indent=4)

# Sauvegarde des prédictions
predictions = pd.DataFrame({"y_test": y_test.values, "y_pred": y_pred})
predictions.to_csv("data/processed_data/predictions.csv", index=False)

print("Evaluation terminée !")
