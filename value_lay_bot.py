import pandas as pd
import pickle
import time
import telegram

# === CONFIGURA QUI ===
BOT_TOKEN = "7359337286:AAFmojWUP9eCKcDLNj5YFb0h_LjJuhjf5uE"
CHAT_ID = "6146221712"
VALUE_THRESHOLD = 0.10
LAY_THRESHOLD = -0.15

# === CARICA MODELLO E DATI ===
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("match_data.csv")

# === CALCOLA PROBABILITÀ, VALUE, LAY ===
X = df[['tournament_enc', 'round_enc', 'odds_player1', 'odds_player2']]
df['model_prob'] = model.predict_proba(X)[:, 1]
df['book_prob'] = 1 / df['odds_player1']
df['value'] = (df['model_prob'] * df['odds_player1']) - 1

# === SELEZIONA VALUE BET E LAY SUGGESTION ===
value_bets = df[df['value'] > VALUE_THRESHOLD]
lay_bets = df[df['value'] < LAY_THRESHOLD]

# === COSTRUISCI MESSAGGIO ===
def format_msg(row):
    msg = f"🎾 [{row['tournament']} - {row['round']}]\n"
    msg += f"{row['player1']} vs {row['player2']}\n\n"
    if row['value'] > 0:
        msg += f"💰 Value bet: {row['player1']} @ {row['odds_player1']:.2f}\n"
        msg += f"📊 Probabilità stimata: {row['model_prob']*100:.0f}% → Value: +{row['value']*100:.0f}%\n"
        msg += f"👉 Consiglio: PUNTARE su {row['player1']}\n"
    else:
        msg += f"❌ Lay suggestion:\n"
        msg += f"{row['player1']} @ {row['odds_player1']:.2f}\n"
        msg += f"📉 Probabilità stimata: {row['model_prob']*100:.0f}% → Value: {row['value']*100:.0f}%\n"
        msg += f"👉 Consiglio: BANCARE {row['player1']} su Betfair\n"
    return msg

# === INVIO SU TELEGRAM ===
bot = telegram.Bot(token=BOT_TOKEN)

for _, row in value_bets.iterrows():
    message = format_msg(row)
    bot.send_message(chat_id=CHAT_ID, text=message)
    time.sleep(1)

for _, row in lay_bets.iterrows():
    message = format_msg(row)
    bot.send_message(chat_id=CHAT_ID, text=message)
    time.sleep(1)
