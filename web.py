# ✅ FINAL VERSION: web.py con supporto pagina web + invio Telegram (corretto async)

import os
import pandas as pd
import pickle
from flask import Flask, render_template_string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from telegram import Bot
import asyncio

app = Flask(__name__)

# === CONFIGURA QUI I TUOI DATI TELEGRAM ===
BOT_TOKEN = "7359337286:AAFmojWUP9eCKcDLNj5YFb0h_LjJuhjf5uE"
CHAT_ID = "6146221712"


def send_telegram_message(message):
    try:
        bot = Bot(token=BOT_TOKEN)
        asyncio.run(bot.send_message(chat_id=CHAT_ID, text=message))
    except Exception as e:
        print("Errore invio Telegram:", e)


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
    messages = []

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
        messages.append(f"🎾 {row['Player_1']} vs {row['Player_2']}\n👉 Vincente previsto: {predicted_winner}")

    # Invia a Telegram una sola volta per visita
    send_telegram_message("\n\n".join(messages))

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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
