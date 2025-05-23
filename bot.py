import pandas as pd
import pickle
import telegram

# Token e chat ID forniti
TOKEN = "7359337286:AAFmojWUP9eCKcDLNj5YFb0h_LjJuhjf5uE"
CHAT_ID = 6146221712

# Carica modello e encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Carica dati
df = pd.read_csv("atp_tennis.csv")
df = df[(df["Rank_1"] > 0) & (df["Rank_2"] > 0)]

# Prendi solo i match più recenti (es. ultimi 3)
latest_matches = df.tail(3)

# Prepara messaggio
bot = telegram.Bot(token=TOKEN)
messages = []

for _, row in latest_matches.iterrows():
    rank_diff = row["Rank_2"] - row["Rank_1"]
    surface_code = encoder.transform([row["Surface"]])[0]
    prediction = model.predict([[rank_diff, surface_code]])[0]
    predicted_winner = row["Player_1"] if prediction == 1 else row["Player_2"]
    messages.append(f"🎾 {row['Player_1']} vs {row['Player_2']}
👉 Vincente previsto: {predicted_winner}")

# Invia su Telegram
final_message = "

".join(messages)
bot.send_message(chat_id=CHAT_ID, text=final_message)