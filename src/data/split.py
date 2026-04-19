import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Chargement des données
df = pd.read_csv("data/raw_data/raw.csv")

# Séparation features / cible
X = df.drop(columns=["silica_concentrate", "date"])
y = df["silica_concentrate"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sauvegarde
os.makedirs("data/processed_data", exist_ok=True)
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)

print("Split terminé !")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
