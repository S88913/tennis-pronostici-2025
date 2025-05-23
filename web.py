import os
import pandas as pd
import pickle
from flask import Flask, render_template_string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

def train_model():
    df = pd.read_csv("atp_tennis.csv")
    df = df[(df["Rank_1"] > 0) & (df["Rank_2"] > 0)]
    df["Rank_Diff"] = df["Rank_2"] - df["Rank_1"]
    df["Surface"] = df["Surface"].fillna("Unknown")

    encoder = LabelEncoder()
    df["Surface_Code"] = encoder.fit_transform(df["Surface"])
    df["Target"] = (df["Winner"] == df["Player_1"]).astype(int)

    X = df[["Rank_Diff", "Surface_Code"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

if not os.path.exists("model.pkl") or not os.path.exists("encoder.pkl"):
    train_model()

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    df = pd.read_csv("atp_tennis.csv")
    df = df[(df["Rank_1"] > 0) & (df["Rank_2"] > 0)]
    latest_matches = df.tail(3)

    predictions = []
    for _, row in latest_matches.iterrows():
        rank_diff = row["Rank_2"] - row["Rank_1"]
        surface_code = encoder.transform([row["Surface"]])[0]
        prediction = model.predict([[rank_diff, surface_code]])[0]
        predicted_winner = row["Player_1"] if prediction == 1 else row["Player_2"]
        predictions.append({
            "player1": row["Player_1"],
            "player2": row["Player_2"],
            "winner": predicted_winner
        })

    html = """
    <h2>🎾 Pronostici Tennis (ultimi 3 match)</h2>
    <ul>
    {% for p in predictions %}
        <li><strong>{{ p.player1 }}</strong> vs <strong>{{ p.player2 }}</strong> 👉 <em>Pronostico: {{ p.winner }}</em></li>
    {% endfor %}
    </ul>
    """
    return render_template_string(html, predictions=predictions)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)