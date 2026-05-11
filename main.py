"""
================================================================================
 SmartAlpha Predictor v3.7.0 — The Underwriter Alpha
================================================================================
 Author      : Jhon Gabriel Simarmata (riel)
 Description : Bot analisis saham BEI berbasis Hybrid Detective.
               - ML Track (Majority Vote: LogReg, RF, XGB)
               - Bandar Track (60d CMF + Price Tightness + OBV Slope)
               - Sectoral Heatmap (Rotasi Sektor)
               - Dividend Detector (Pre-run-up & Trap Filter)
               - Multi-Timeframe (Hourly MA20 Confirmation)
               - Underwriter Radar (Broker Power Scoring)
================================================================================
"""

import os
from dotenv import load_dotenv
load_dotenv()

import logging
import asyncio
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from xgboost import XGBClassifier

import telegram

# ──────────────────────────────────────────────────────────────────────────────
# DATABASE & KONFIGURASI
# ──────────────────────────────────────────────────────────────────────────────

# Rating Broker Underwriter (5: Agresif/Market Maker, 3: BUMN/Konservatif)
BROKER_POWER = {
    "YP": 5, "MG": 5, "XC": 4, "PD": 4, "NI": 3, "DH": 3, "OD": 2, "AZ": 4, "LG": 4
}

CONFIG = {
    "TELEGRAM_TOKEN":   os.getenv("TELEGRAM_TOKEN"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
    "MIN_PRICE": 100,
    "MAX_PRICE": 15000,
    "MIN_VOLUME": 1_000_000,
    "MIN_WIN_PROB": 0.53,
    "CMF_LONG": 60,
    "TIGHTNESS_THRESHOLD": 0.03, # 3% Max StdDev
    "STOCK_UNIVERSE": [
        # Contoh pemetaan metadata. Tambahkan daftar 275 sahammu di sini.
        {"ticker": "BBCA.JK", "sector": "Perbankan", "uw": "PD", "div": [4, 12]},
        {"ticker": "BBRI.JK", "sector": "Perbankan", "uw": "PD", "div": [3, 12]},
        {"ticker": "BJTM.JK", "sector": "Perbankan", "uw": "NI", "div": [5]},
        {"ticker": "BJBR.JK", "sector": "Perbankan", "uw": "PD", "div": [5]},
        {"ticker": "ADRO.JK", "sector": "Energi", "uw": "DH", "div": [5, 12]},
        {"ticker": "PTBA.JK", "sector": "Energi", "uw": "OD", "div": [6]},
        {"ticker": "ITMG.JK", "sector": "Energi", "uw": "DH", "div": [4, 9]},
        {"ticker": "GOTO.JK", "sector": "Tech", "uw": "YP", "div": []},
        {"ticker": "WIFI.JK", "sector": "Tech", "uw": "MG", "div": []},
        {"ticker": "ACES.JK", "sector": "Konsumer", "uw": "XC", "div": [5]},
        {"ticker": "KAEF.JK", "sector": "Healthcare", "uw": "NI", "div": [5]},
        {"ticker": "SMRA.JK", "sector": "Properti", "uw": "LG", "div": [6]},
        {"ticker": "BRIS.JK", "sector": "Perbankan", "uw": "NI", "div": [5]},
        {"ticker": "ISAT.JK", "sector": "Telco", "uw": "DH", "div": [5]},
    ],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 1: INDIKATOR TEKNIKAL & BANDARMOLOGI
# ══════════════════════════════════════════════════════════════════════════════

def _calc_cmf(df, period=20):
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    return (mfm * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()

def _calc_indicators(df):
    d = df.copy()
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["RSI"] = 50 # Placeholder, implement standard RSI if needed
    d["CMF20"] = _calc_cmf(d, 20)
    d["CMF60"] = _calc_cmf(d, 60)
    d["Tightness"] = d["Close"].rolling(20).std() / d["Close"].rolling(20).mean()
    d["OBV"] = (np.sign(d["Close"].diff().fillna(0)) * d["Volume"]).cumsum()
    d["Target"] = (d["Close"].shift(-1) > d["Close"]).astype(int)
    return d

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 2: TRACK ANALISIS (ML, BANDAR, DIV, MTF, UW)
# ══════════════════════════════════════════════════════════════════════════════

async def analyze_stock(item):
    t = item["ticker"]
    try:
        df_d = yf.download(t, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df_d.empty or len(df_d) < 70: return None
        if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
        
        df = _calc_indicators(df_d)
        last = df.iloc[-1]
        
        # --- TRACK 1: ML (Simplified Majority Vote) ---
        feats = ["Close", "Volume", "MA20", "CMF20", "Tightness"]
        df_train = df.iloc[:-1].dropna()
        X, y = df_train[feats].values, df_train["Target"].values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        
        lr = LogisticRegression().fit(X_sc, y)
        rf = RandomForestClassifier(n_estimators=50).fit(X_sc, y)
        xgb = XGBClassifier(eval_metric='logloss').fit(X_sc, y)
        
        last_sc = scaler.transform(df[feats].tail(1).values)
        votes = [lr.predict(last_sc)[0], rf.predict(last_sc)[0], xgb.predict(last_sc)[0]]
        win_prob = (lr.predict_proba(last_sc)[0][1] + rf.predict_proba(last_sc)[0][1] + xgb.predict_proba(last_sc)[0][1]) / 3

        # --- TRACK 2: Bandar (60d Accumulation) ---
        is_accum = (last["CMF60"] > 0.05) and (last["Tightness"] < CONFIG["TIGHTNESS_THRESHOLD"])
        
        # --- TRACK 3: Dividend Detector ---
        now_m = datetime.now().month
        is_div_season = now_m in item["div"] or ((now_m % 12) + 1) in item["div"]
        div_trap = is_div_season and (last["Close"] / df["Close"].iloc[-45] - 1 > 0.15)

        # --- TRACK 4: Multi-Timeframe ---
        df_h = yf.download(t, period="5d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(df_h.columns, pd.MultiIndex): df_h.columns = df_h.columns.get_level_values(0)
        mtf_bullish = df_h["Close"].iloc[-1] > df_h["Close"].rolling(20).mean().iloc[-1] if not df_h.empty else False

        # --- TRACK 5: Underwriter Radar ---
        uw_name = item.get("uw", "Unknown")
        uw_score = BROKER_POWER.get(uw_name, 0)
        is_uw_active = (uw_score >= 4 and last["CMF60"] > 0.08)

        return {
            "ticker": t, "price": last["Close"], "sector": item["sector"],
            "win_prob": win_prob, "agree_count": sum(votes), "passed_ml": (sum(votes) >= 2 and win_prob > CONFIG["MIN_WIN_PROB"]),
            "cmf60": last["CMF60"], "is_accum": is_accum, "tightness": last["Tightness"],
            "div_status": "🔥 Pre-Div" if (is_div_season and not div_trap) else ("⚠️ Trap" if div_trap else "—"),
            "mtf_ok": mtf_bullish, "uw": uw_name, "uw_score": uw_score, "uw_active": is_uw_active
        }
    except Exception as e:
        logger.error(f"Error {t}: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 3: FORMATTER & PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def format_master_dashboard(results):
    now = datetime.now().strftime("%d %b %Y, %H:%M WIB")
    vip = [r for r in results if r["passed_ml"]]
    
    msg_list = []
    # Header & ML
    h = [f"🤖 *SmartAlpha v3.7.0*\n📅 {now}\n" + "━"*20 + "\n🏆 *TOP ML SIGNALS*"]
    if not vip: h.append("`Nol sinyal ML hari ini.`")
    else:
        for i, r in enumerate(sorted(vip, key=lambda x: x['win_prob'], reverse=True)[:5], 1):
            h.append(f"{i}. *{r['ticker'].replace('.JK','')}* Prob: `{r['win_prob']:.1%}` | MTF: {'✅' if r['mtf_ok'] else '❌'}")
    msg_list.append("\n".join(h))

    # Bandar & Underwriter
    b = ["🏛️ *UNDERWRITER & BANDAR RADAR*"]
    for i, r in enumerate(sorted(results, key=lambda x: (x['uw_score'], x['cmf60']), reverse=True)[:8], 1):
        star = "🔥" * r['uw_score']
        status = "🐳" if r['uw_active'] else "⚪"
        b.append(f"{i}. *{r['ticker'].replace('.JK','')}* ({r['uw']}) {star}\n   CMF60: `{r['cmf60']:+.3f}` | {status} {r['div_status']}")
    msg_list.append("\n".join(b))

    # Heatmap Sektor
    df_res = pd.DataFrame(results)
    heatmap = df_res.groupby('sector')['cmf60'].mean().sort_values(ascending=False).to_dict()
    s = ["🗺️ *SECTORAL HEATMAP (Avg CMF60)*"]
    for sect, val in list(heatmap.items())[:5]:
        s.append(f"{'🔥' if val > 0.05 else '❄️'} `{sect:<12}`: `{val:+.3f}`")
    msg_list.append("\n".join(s))

    return msg_list

async def run_pipeline():
    logger.info("Pipeline v3.7 Starting...")
    results = []
    for item in CONFIG["STOCK_UNIVERSE"]:
        res = await analyze_stock(item)
        if res: results.append(res)
        await asyncio.sleep(0.5)
    
    if results:
        bot = telegram.Bot(token=CONFIG["TELEGRAM_TOKEN"])
        messages = format_master_dashboard(results)
        for m in messages:
            await bot.send_message(chat_id=CONFIG["TELEGRAM_CHAT_ID"], text=m, parse_mode="Markdown")
            await asyncio.sleep(1)
    logger.info("Pipeline Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
