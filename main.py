"""
================================================================================
 BEI Stock Trend Classifier & Probability Detector
================================================================================
 Author      : Jhon Gabriel Simarmata (riel)
 Version     : 3.2.0 "The Ultimate Classifier"
 Description : Bot otomatis klasifikasi arah tren saham BEI.
               - Mode Majority Vote (Minimal 2 dari 3 model setuju)
               - Radar Bandar (CMF/OBV) Always On
               - Anti-Falling Knife & Anti-Suspend Filters
               - Fix Data Leakage pada Target Variable
================================================================================
"""

import os
from dotenv import load_dotenv
load_dotenv()

import logging
import asyncio
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score
from xgboost import XGBClassifier

import telegram

# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURASI UTAMA
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # --- Telegram ---
    "TELEGRAM_TOKEN":   os.getenv("TELEGRAM_TOKEN",  "YOUR_TELEGRAM_TOKEN_HERE"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE"),

    # --- Screener ---
    "MIN_PRICE_PER_SHARE": 200,
    "MAX_PRICE_PER_SHARE": 1_000,
    "MIN_AVG_VOLUME":      1_000_000,

    # --- Machine Learning ---
    "LOOKBACK_DAYS":      60,
    "TEST_SIZE":          0.2,

    # LogisticRegression
    "LR_MAX_ITER":        1_000,
    "LR_RANDOM_STATE":    42,

    # RandomForestClassifier
    "RF_N_ESTIMATORS":    100,
    "RF_RANDOM_STATE":    42,

    # XGBClassifier
    "XGB_N_ESTIMATORS":   100,
    "XGB_LEARNING_RATE":  0.05,
    "XGB_MAX_DEPTH":      4,
    "XGB_RANDOM_STATE":   42,

    # [V3.2] Confidence Filter
    "MIN_WIN_PROBABILITY": 0.53,

    # --- Indikator Teknikal ---
    "CMF_PERIOD": 20,
    "BB_PERIOD":  20,
    "BB_STD":     2.0,
    "STOCH_K":    14,
    "STOCH_D":    3,

    # --- Watchlist Saham BEI (275 Active Stocks) ---
    "STOCK_TICKERS": [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "ARTO.JK", "BBYB.JK", 
        "AGRO.JK", "BBTN.JK", "PNLF.JK", "PNBN.JK", "BDMN.JK", "BNGA.JK", "NISP.JK", 
        "BJBR.JK", "BJTM.JK", "BTPS.JK", "BTPN.JK", "BNLI.JK", "MEGA.JK", "MAYA.JK",
        "BNBA.JK", "MCOR.JK", "NOBU.JK", "INPC.JK", "SDRA.JK", "BGTG.JK", "AMAR.JK",
        "TRIN.JK", "VINS.JK", "CFIN.JK", "BFIN.JK", "WOMF.JK", "ADMF.JK", "MFIN.JK",
        "ADRO.JK", "PTBA.JK", "ITMG.JK", "HRUM.JK", "BUMI.JK", "BRMS.JK", "DEWA.JK",
        "ENRG.JK", "MEDC.JK", "ELSA.JK", "PGAS.JK", "AKRA.JK", "INDY.JK", "DOID.JK",
        "MBAP.JK", "TOBA.JK", "KKGI.JK", "BSSR.JK", "GEMS.JK", "SMMT.JK", "ABMM.JK",
        "BIPI.JK", "PGEO.JK", "KEEN.JK", "ARKO.JK", "RAJA.JK", "WINS.JK", "LEAD.JK",
        "SOCI.JK", "CARS.JK", "MCOL.JK", "TCPI.JK", "TPMA.JK", "BSML.JK", "GTBO.JK",
        "ANTM.JK", "INCO.JK", "MDKA.JK", "TINS.JK", "PSAB.JK", "DKFT.JK", "AMMN.JK",
        "NCKL.JK", "MBMA.JK", "NCCC.JK", "ZINC.JK", "IFSH.JK", "CITA.JK", "SURE.JK",
        "BSDE.JK", "CTRA.JK", "SMRA.JK", "PWON.JK", "APLN.JK", "ASRI.JK", "DILD.JK",
        "KIJA.JK", "LPCK.JK", "LPKR.JK", "BEST.JK", "GWSA.JK", "CITY.JK", "MDLN.JK",
        "MTLA.JK", "JRPT.JK", "GPRA.JK", "DART.JK", "OMRE.JK", "NIRO.JK", "BKSL.JK",
        "PANI.JK", "MKPI.JK", "RODA.JK", "BAPA.JK", "MMLP.JK", "SSIA.JK", "DMAS.JK",
        "WIKA.JK", "PTPP.JK", "ADHI.JK", "WSKT.JK", "WEGE.JK", "WTON.JK", "PPRE.JK",
        "NRCA.JK", "TOTL.JK", "ACST.JK", "META.JK", "CMNP.JK", "JSMR.JK", "CSIS.JK",
        "IDPR.JK", "PORT.JK", "IPCC.JK", "IPCM.JK", "NELY.JK", "HAIS.JK", "CASS.JK",
        "GOTO.JK", "BUKA.JK", "BELI.JK", "WIFI.JK", "TLKM.JK", "ISAT.JK", "EXCL.JK",
        "FREN.JK", "TOWR.JK", "TBIG.JK", "MTEL.JK", "EMTK.JK", "SCMA.JK", "MNCN.JK",
        "BMTR.JK", "MSIN.JK", "MLPT.JK", "MTDL.JK", "ASRM.JK", "DNET.JK", "KREN.JK",
        "MCAS.JK", "NFCX.JK", "TFAS.JK", "MARI.JK", "ABBA.JK", "VIVA.JK", "FILM.JK",
        "INDF.JK", "ICBP.JK", "MYOR.JK", "UNVR.JK", "KLBF.JK", "SIDO.JK", "ROTI.JK",
        "CAMP.JK", "GOOD.JK", "CLEO.JK", "AMRT.JK", "MIDI.JK", "ERAA.JK", "MAPI.JK",
        "MAPA.JK", "ACES.JK", "RALS.JK", "LPPF.JK", "WOOD.JK", "KINO.JK", "WISM.JK",
        "HZLN.JK", "CMRY.JK", "STAA.JK", "PCAR.JK", "CINT.JK", "TCID.JK", "MBTO.JK",
        "KAEF.JK", "PEHA.JK", "TSPC.JK", "DVLA.JK", "SCPI.JK", "INAF.JK", "GGRM.JK",
        "HMSP.JK", "WIIM.JK", "RMBA.JK", "MLBI.JK", "DLTA.JK",
        "CPIN.JK", "JPFA.JK", "MAIN.JK", "WMUU.JK", "AALI.JK", "LSIP.JK", "DSNG.JK",
        "SIMP.JK", "SSMS.JK", "BWPT.JK", "TBLA.JK", "ANJT.JK", "TAPG.JK", "SMAR.JK",
        "PALM.JK", "BISI.JK", "CPRO.JK", "SGRO.JK", "JAWA.JK", "CSRA.JK", "FAPA.JK",
        "MGRO.JK", "BTEK.JK",
        "ASII.JK", "AUTO.JK", "GJTL.JK", "SMSM.JK", "SMGR.JK", "INTP.JK", "SMBR.JK",
        "TKIM.JK", "INKP.JK", "BRPT.JK", "TPIA.JK", "ESSA.JK", "ARNA.JK", "MARK.JK",
        "AVIA.JK", "IMAS.JK", "DRMA.JK", "LPIN.JK", "SMCB.JK", "FASW.JK", "SPMA.JK",
        "ALDO.JK", "KDSI.JK", "BRNA.JK", "IGAR.JK", "TRST.JK", "YPAS.JK", "LION.JK",
        "SRSN.JK", "AGII.JK", "MOLI.JK", "DPNS.JK", "BUDI.JK", "DPUM.JK",
        "MIKA.JK", "SILO.JK", "HEAL.JK", "PRDA.JK", "SAME.JK", "IRRA.JK", "CARE.JK",
        "BMHS.JK", "RSGK.JK", "SRAJ.JK", "TMAS.JK", "SMDR.JK", "BIRD.JK", "ASLC.JK",
        "ASSA.JK", "TRUK.JK", "SAPX.JK"
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# SETUP LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 1: DATA ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def get_stock_data(ticker: str, period_days: int = 90) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=f"{period_days}d", progress=False, auto_adjust=True)
        if df.empty or len(df) < (period_days * 0.5):
            logger.warning(f"[{ticker}] Data tidak cukup ({len(df)} baris).")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            logger.warning(f"[{ticker}] Kolom OHLCV tidak lengkap.")
            return None

        df.dropna(how="all", inplace=True)
        return df

    except Exception as e:
        logger.error(f"[{ticker}] Gagal mengambil data: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 2: SCREENER
# ══════════════════════════════════════════════════════════════════════════════

def screen_stocks(tickers: list[str]) -> list[dict]:
    logger.info("=" * 60)
    logger.info(f"Screening {len(tickers)} saham BEI...")
    logger.info("=" * 60)

    passed = []
    for ticker in tickers:
        time.sleep(0.5) 

        df = get_stock_data(ticker, period_days=30)
        if df is None:
            continue

        ma20_series = df["Close"].rolling(window=20, min_periods=10).mean().dropna()
        if ma20_series.empty:
            continue

        current_price = float(df["Close"].iloc[-1])
        ma20_current  = float(ma20_series.iloc[-1])
        avg_volume    = float(df["Volume"].tail(20).mean())
        price_per_lot = current_price * 100

        # Filter 1 & 2: Harga & Volume
        min_p, max_p = CONFIG["MIN_PRICE_PER_SHARE"], CONFIG["MAX_PRICE_PER_SHARE"]
        if not (min_p <= current_price <= max_p) or (avg_volume < CONFIG["MIN_AVG_VOLUME"]):
            continue

        # Filter 3: Anti-Falling Knife
        if current_price <= ma20_current:
            continue
            
        # Filter 4: Anti-Suspend / Saham Tidur
        recent_volume = df["Volume"].tail(2).sum()
        recent_price_std = df["Close"].tail(5).std()
        if recent_volume == 0 or recent_price_std == 0.0:
            logger.info(f"  ✗ {ticker} SUSPEND: Volume mati atau harga stagnan.")
            continue

        logger.info(f"  ✓ LOLOS | Rp{current_price:,.0f}/lbr | Vol {avg_volume:,.0f}")
        passed.append({
            "ticker": ticker,
            "current_price": current_price,
            "price_per_lot": price_per_lot,
            "avg_volume": avg_volume,
        })

    logger.info(f"\nScreening: {len(passed)}/{len(tickers)} lolos.\n")
    return passed

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 3: INDIKATOR TEKNIKAL
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line, macd_line.ewm(span=signal, adjust=False).mean()

def _calc_bollinger_bands(close: pd.Series, period: int = 20, n_std: float = 2.0):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return sma + (n_std * std), sma - (n_std * std)

def _calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    price_range = (highest_high - lowest_low).replace(0, np.nan)
    pct_k = ((close - lowest_low) / price_range) * 100
    return pct_k, pct_k.rolling(window=d_period).mean()

def _calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()

def _calc_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    return (mfm * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

FEATURE_COLS = [
    "Close", "Volume", "MA5", "MA20", "MA_Signal", "Price_Change",
    "RSI14", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower",
    "Stoch_K", "Stoch_D", "OBV", "CMF",
]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA_Signal"] = df["MA5"] - df["MA20"]
    df["Price_Change"] = df["Close"].pct_change() * 100
    df["RSI14"] = _calc_rsi(df["Close"], period=14)
    df["MACD"], df["MACD_Signal"] = _calc_macd(df["Close"])
    df["Stoch_K"], df["Stoch_D"] = _calc_stochastic(df["High"], df["Low"], df["Close"])
    df["BB_Upper"], df["BB_Lower"] = _calc_bollinger_bands(df["Close"])
    df["OBV"] = _calc_obv(df["Close"], df["Volume"])
    df["CMF"] = _calc_cmf(df["High"], df["Low"], df["Close"], df["Volume"])

    future_close = df["Close"].shift(-1)
    df["Target"] = np.where(future_close > df["Close"], 1, 0)
    df.iloc[-1, df.columns.get_loc("Target")] = np.nan
    
    return df

def generate_ai_analysis(last_row: pd.Series) -> str:
    def safe(key, default=np.nan):
        v = last_row.get(key, default)
        return v if pd.notna(v) else default

    parts = []
    close, ma20 = safe("Close"), safe("MA20")
    if pd.notna(close) and pd.notna(ma20):
        parts.append("Uptrend ↑" if close > ma20 else "Sideways/Down ↓")
    
    rsi = safe("RSI14")
    if pd.notna(rsi):
        rsi_i = int(round(rsi))
        parts.append(f"RSI OB ({rsi_i})⚠️" if rsi >= 70 else f"RSI OS ({rsi_i})🔥" if rsi <= 30 else f"RSI Netral ({rsi_i})")
    
    cmf = safe("CMF")
    if pd.notna(cmf):
        parts.append("Akumulasi 🐳" if cmf > 0.1 else "Distribusi ⚠️" if cmf < -0.1 else "Vol Netral")
    
    stoch_k = safe("Stoch_K")
    if pd.notna(stoch_k):
        if stoch_k >= 80: parts.append("Stoch OB")
        elif stoch_k <= 20: parts.append("Stoch OS 🔥")

    return ", ".join(parts)

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 4: ML CLASSIFIER & MAJORITY VOTE
# ══════════════════════════════════════════════════════════════════════════════

def train_and_predict(ticker: str) -> dict | None:
    logger.info(f"[{ticker}] Training Tribrid Classifier...")
    df_raw = get_stock_data(ticker, period_days=CONFIG["LOOKBACK_DAYS"] + 30)
    if df_raw is None: return None

    df = create_features(df_raw)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing: return None

    live_data = df.iloc[-1:]
    df_train = df.iloc[:-1].dropna()

    if len(df_train) < 20: return None
    X, y = df_train[FEATURE_COLS].values, df_train["Target"].values
    if len(np.unique(y)) < 2: return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["TEST_SIZE"], shuffle=False)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2: return None

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=CONFIG["LR_MAX_ITER"], random_state=CONFIG["LR_RANDOM_STATE"])
    rf = RandomForestClassifier(n_estimators=CONFIG["RF_N_ESTIMATORS"], random_state=CONFIG["RF_RANDOM_STATE"], n_jobs=-1)
    xgb = XGBClassifier(n_estimators=CONFIG["XGB_N_ESTIMATORS"], learning_rate=CONFIG["XGB_LEARNING_RATE"], max_depth=CONFIG["XGB_MAX_DEPTH"], random_state=CONFIG["XGB_RANDOM_STATE"], tree_method="hist", verbosity=0, eval_metric="logloss")
    
    lr.fit(X_train_sc, y_train)
    rf.fit(X_train_sc, y_train)
    xgb.fit(X_train_sc, y_train)

    ensemble_test = ((lr.predict(X_test_sc) + rf.predict(X_test_sc) + xgb.predict(X_test_sc)) >= 2).astype(int)
    precision = precision_score(y_test, ensemble_test, zero_division=0)
    accuracy  = accuracy_score(y_test, ensemble_test)

    last_X = live_data[FEATURE_COLS].values
    last_sc = scaler.transform(last_X)

    lr_pred, rf_pred, xgb_pred = int(lr.predict(last_sc)[0]), int(rf.predict(last_sc)[0]), int(xgb.predict(last_sc)[0])
    
    votes = {"LogReg": lr_pred, "RF": rf_pred, "XGB": xgb_pred}
    agree_count = sum(1 for v in votes.values() if v == 1)

    passed_ml = True
    if agree_count < 2:
        logger.info(f"[{ticker}] ✗ CONFIDENCE: Hanya {agree_count}/3 model vote Naik.")
        passed_ml = False
    else:
        logger.info(f"[{ticker}] ✓ CONFIDENCE: {agree_count}/3 model vote Naik 🟢")

    lr_prob  = float(lr.predict_proba(last_sc)[0][1])
    rf_prob  = float(rf.predict_proba(last_sc)[0][1])
    xgb_prob = float(xgb.predict_proba(last_sc)[0][1])
    win_probability = (lr_prob + rf_prob + xgb_prob) / 3.0

    if win_probability < CONFIG["MIN_WIN_PROBABILITY"]:
        logger.info(f"[{ticker}] ✗ LOW PROB: {win_probability:.1%} < {CONFIG['MIN_WIN_PROBABILITY']:.0%}")
        passed_ml = False

    last_row = live_data.iloc[-1]
    def safe_float(key, default=0.0):
        val = last_row.get(key, np.nan)
        try:
            return float(val) if np.isfinite(float(val)) else default
        except:
            return default

    return {
        "ticker": ticker,
        "current_price": safe_float("Close"),
        "price_per_lot": safe_float("Close") * 100,
        "win_probability": win_probability,
        "lr_prob": lr_prob, "rf_prob": rf_prob, "xgb_prob": xgb_prob,
        "precision": precision, "accuracy": accuracy,
        "cmf_current": safe_float("CMF"), "obv_current": safe_float("OBV"),
        "ai_analysis": generate_ai_analysis(last_row),
        "passed_ml": passed_ml,
        "agree_count": agree_count
    }

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 5: TELEGRAM FORMATTER & SENDER
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_obv(obv: float) -> str:
    a = abs(obv)
    if a >= 1_000_000_000: return f"{obv / 1_000_000_000:+.2f}M lot"
    elif a >= 1_000_000: return f"{obv / 1_000_000:+.1f}Jt"
    return f"{obv:+,.0f}"

def format_telegram_message(results: list[dict]) -> str:
    now = datetime.now().strftime("%d %b %Y, %H:%M WIB")
    vip_results = [r for r in results if r.get("passed_ml", False)]
    lines = [
        "🤖 *BEI Trend Classifier v3.2*",
        f"📅 {now}   |   🧠 LogReg + RF + XGBoost",
        f"🎯 Mode: Majority Vote (2/3) + CMF Radar",
        f"📊 {len(vip_results)} saham lolos ML Filter (dari {len(results)} screened)",
        "━" * 36, "", "🏆 *TOP 15 — PROBABILITAS NAIK BESOK*", ""
    ]

    if not vip_results:
        lines += ["⚠️ Tidak ada saham yang lolos filter Machine Learning.", "Semua probabilitas di bawah batas. Cek Radar Bandar di bawah!\n"]
    else:
        for i, r in enumerate(sorted(vip_results, key=lambda x: x.get("win_probability", 0), reverse=True)[:15], 1):
            lines += [
                f"🟢 *{i}. {r['ticker'].replace('.JK', '')}*",
                f"   Harga/Lot   : Rp{r.get('price_per_lot', 0):>9,.0f}",
                f"   Prob. Naik  : *{r.get('win_probability',0):.1%}* ({r.get('agree_count',0)}/3 Models Agree)",
                f"   Precision   : {r.get('precision', 0):.1%} (Keandalan Sinyal)",
                f"   🤖 _{r.get('ai_analysis', 'N/A')}_", ""
            ]

    lines += ["━" * 36, "", "🐳 *RADAR AKUMULASI — Top 5 Bandar Masuk*", "_(CMF positif tertinggi = tekanan beli dominan)_", ""]
    top_accum = sorted(results, key=lambda x: x.get("cmf_current", -999), reverse=True)[:5]
    for i, r in enumerate(top_accum, 1):
        lines.append(f"  {i}. *{r['ticker'].replace('.JK',''):<6}* CMF `{r.get('cmf_current',0):+.3f}` | OBV `{_fmt_obv(r.get('obv_current',0))}`")

    lines += ["", "━" * 36, "", "⚠️ *RADAR DISTRIBUSI — Top 5 Bandar Keluar*", "_(CMF negatif terdalam = tekanan jual dominan)_", ""]
    top_distrib = sorted(results, key=lambda x: x.get("cmf_current", 999))[:5]
    for i, r in enumerate(top_distrib, 1):
        lines.append(f"  {i}. *{r['ticker'].replace('.JK',''):<6}* CMF `{r.get('cmf_current',0):+.3f}` | OBV `{_fmt_obv(r.get('obv_current',0))}`")

    lines += ["", "━" * 36, "⚠️ Disclaimer: Edukasi & riset. Bukan ajakan beli/jual."]
    return "\n".join(lines)

async def send_telegram_message(message: str) -> bool:
    token, chat_id = CONFIG["TELEGRAM_TOKEN"], CONFIG["TELEGRAM_CHAT_ID"]
    if "YOUR_TELEGRAM_TOKEN_HERE" in token: return False
    bot = telegram.Bot(token=token)
    try:
        for i in range(0, len(message), 4000):
            await bot.send_message(chat_id=chat_id, text=message[i:i+4000], parse_mode="Markdown")
            await asyncio.sleep(0.5)
        return True
    except Exception as e:
        logger.error(f"Gagal mengirim Telegram: {e}")
        return False

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 6: PIPELINE UTAMA
# ══════════════════════════════════════════════════════════════════════════════

async def run_pipeline():
    logger.info("🚀 BEI Trend Classifier v3.2 — Pipeline dimulai...")
    t0 = time.time()
    tickers = CONFIG["STOCK_TICKERS"]
    screened = screen_stocks(tickers)
    
    if not screened: return
    
    results = [pred for stock in screened if (pred := train_and_predict(stock["ticker"]))]
    if results: await send_telegram_message(format_telegram_message(results))
    logger.info(f"\n✅ Pipeline selesai dalam {time.time() - t0:.1f} detik.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())