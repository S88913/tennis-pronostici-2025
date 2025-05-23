# Tennis Pronostici Bot 🇮🇹

Questo progetto usa dati ATP + un modello di Machine Learning per prevedere il vincitore dei match recenti e inviare pronostici su Telegram.

## Cosa fa:
- Prepara un modello semplice su base storico (RandomForest)
- Predice chi vincerà (Player_1 o Player_2)
- Invia i pronostici via Telegram (privato)

## Istruzioni:
1. Clona il repo
2. Inserisci `atp_tennis.csv` nella root
3. `python model.py` per allenare il modello
4. `python bot.py` per ricevere i pronostici

Puoi anche deployare su [Render](https://render.com) automaticamente.