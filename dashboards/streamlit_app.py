import os, pandas as pd, joblib, streamlit as st
from sklearn.linear_model import LogisticRegression
from ingestion.fetch_binance import fetch_binance           # already in your repo
from features.core import generate_features                # already in your repo

DATA_FILE  = "data/processed/BTCUSDT_features.parquet"
MODEL_FILE = "models/logistic_model.pkl"

# ----------------------------------------------------------------------
#  First-run bootstrap: fetch data ➜ create features ➜ train model
# ----------------------------------------------------------------------
if not os.path.exists(DATA_FILE):
    st.info("⏬ First run: downloading 1 000 one-minute bars from Binance…")
    df_raw   = fetch_binance("BTCUSDT", interval="1m", limit=1000)
    os.makedirs("data/processed", exist_ok=True)
    df_feat  = generate_features(df_raw)
    df_feat.to_parquet(DATA_FILE, index=False)

if not os.path.exists(MODEL_FILE):
    st.info("🧠 Training logistic model …")
    df = pd.read_parquet(DATA_FILE)
    df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)
    df.dropna(inplace=True)

    X = df[["zscore", "rsi", "vol"]]
    y = df["target"]
    model = LogisticRegression().fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_FILE)
else:
    model = joblib.load(MODEL_FILE)

# ----------------------------------------------------------------------
#  Dashboard
# ----------------------------------------------------------------------
st.title("ChronosEdge — BTC 5-Minute Upside Probability")

df = pd.read_parquet(DATA_FILE)
df["prob_up"] = model.predict_proba(df[["zscore", "rsi", "vol"]])[:, 1]

tab1, tab2 = st.tabs(["Price vs probability", "Raw feature table"])

with tab1:
    st.line_chart(
        df.set_index("open_time")[["close", "prob_up"]],
        height=350,
    )

with tab2:
    st.dataframe(df.tail(200), use_container_width=True)
