import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Carica il file dati
df = pd.read_csv("atp_tennis.csv")

# Rimuovi righe con dati mancanti o quote placeholder
df = df[(df["Rank_1"] > 0) & (df["Rank_2"] > 0)]

# Feature engineering: ranking difference
df["Rank_Diff"] = df["Rank_2"] - df["Rank_1"]
df["Surface"] = df["Surface"].fillna("Unknown")

# Encoding della superficie
surface_encoder = LabelEncoder()
df["Surface_Code"] = surface_encoder.fit_transform(df["Surface"])

# Target: vincitore (1 se Player_1 vince, 0 altrimenti)
df["Target"] = (df["Winner"] == df["Player_1"]).astype(int)

# Selezione delle feature
X = df[["Rank_Diff", "Surface_Code"]]
y = df["Target"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Salva modello e codificatore
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoder.pkl", "wb") as f:
    pickle.dump(surface_encoder, f)