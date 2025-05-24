import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("match_data.csv")
X = df[['tournament_enc', 'round_enc', 'odds_player1', 'odds_player2']]
y = df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modello salvato come model.pkl")
