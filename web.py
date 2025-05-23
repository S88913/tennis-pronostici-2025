from flask import Flask, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)

# Carica il modello e l'encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    # Carica i dati
    df = pd.read_csv("atp_tennis.csv")
    df = df[(df["Rank_1"] > 0) & (df["Rank_2"] > 0)]

    # Prendi gli ultimi 3 match
    latest_matches = df.tail(3)

    # Genera i pronostici
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

    # HTML molto semplice
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