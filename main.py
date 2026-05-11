"""
================================================================================
 SmartAlpha Predictor — BEI Hybrid Detective Bot
================================================================================
 Author      : Jhon Gabriel Simarmata (riel)
 Version     : 3.6.0 "Master Alpha"
 Description : Bot otomatis analisis saham BEI dengan filosofi "Hybrid
               Detective" — menggabungkan ML Track dan Bandar Track untuk
               mendeteksi akumulasi institusional sebelum pergerakan besar.

 ── ARSITEKTUR V3.6 ──────────────────────────────────────────────────────────

 [TRACK 1] MACHINE LEARNING
   Tribrid Ensemble: Logistic Regression + Random Forest + XGBoost
   Voting Rule: Minimal 2/3 model sepakat "Naik" (Majority Vote)
   Filter: Win Probability ≥ 53%

 [TRACK 2] BANDAR TRACK (PRIORITAS UTAMA)
   ├─ Long-Term CMF (60d)  : Deteksi akumulasi senyap institusional
   ├─ Price Tightness       : StdDev Close 20d < 3% → bandar "jaga" harga
   └─ OBV Trend             : Konfirmasi arah volume kumulatif

 [TRACK 3] SECTORAL HEATMAP
   Rata-rata CMF per sektor → deteksi rotasi sektor & institutional inflow

 [TRACK 4] DIVIDEND DETECTOR
   Pre-Dividend Run-up: Deteksi CMF spike 1-2 bulan sebelum Cum-Date historis
   Dividend Trap Filter: Hindari saham yang sudah terlalu naik jelang dividen

 [TRACK 5] MULTI-TIMEFRAME CONFIRMATION
   Daily signal dikonfirmasi oleh Hourly MA20 → hindari fakeout

 ── DEPLOYMENT ────────────────────────────────────────────────────────────────
   GitHub Actions: Scheduled daily 16:10 WIB (09:10 UTC, Senin–Jumat)

 ── CHANGELOG V3.2 → V3.6 ───────────────────────────────────────────────────
   + Bandar Track: CMF 60d window, Price Tightness filter
   + Sectoral Heatmap dengan rata-rata CMF per sektor
   + Dividend Detector: Pre-run-up scanner + Trap filter
   + Multi-Timeframe: konfirmasi Hourly MA20
   + Telegram Dashboard dirombak: 5 seksi terpisah
   ~ Anti-Suspend filter dipertahankan dari V3.2
   ~ Data leakage fix (live_data/df_train split) dipertahankan
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# load_dotenv() SEBELUM os.getenv() manapun — urutan ini wajib.
# ──────────────────────────────────────────────────────────────────────────────

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
from sklearn.metrics import precision_score, accuracy_score
from xgboost import XGBClassifier

import telegram


# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURASI UTAMA
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Telegram ──────────────────────────────────────────────────────────────
    "TELEGRAM_TOKEN":   os.getenv("TELEGRAM_TOKEN",  "YOUR_TELEGRAM_TOKEN_HERE"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE"),

    # ── Screener ──────────────────────────────────────────────────────────────
    "MIN_PRICE_PER_SHARE": 200,
    "MAX_PRICE_PER_SHARE": 1_000,
    "MIN_AVG_VOLUME":      1_000_000,

    # ── ML Track ─────────────────────────────────────────────────────────────
    "LOOKBACK_DAYS":       60,
    "TEST_SIZE":           0.2,
    "LR_MAX_ITER":         1_000,
    "LR_RANDOM_STATE":     42,
    "RF_N_ESTIMATORS":     100,
    "RF_RANDOM_STATE":     42,
    "XGB_N_ESTIMATORS":    100,
    "XGB_LEARNING_RATE":   0.05,
    "XGB_MAX_DEPTH":       4,
    "XGB_RANDOM_STATE":    42,
    "MIN_WIN_PROBABILITY": 0.53,

    # ── Indikator Teknikal (Daily) ────────────────────────────────────────────
    "CMF_PERIOD":     20,    # CMF reguler untuk ML feature
    "CMF_LONG_DAYS":  60,    # [V3.6] CMF jangka panjang untuk Bandar Track
    "BB_PERIOD":      20,
    "BB_STD":         2.0,
    "STOCH_K":        14,
    "STOCH_D":        3,

    # ── [V3.6] Bandar Track ───────────────────────────────────────────────────
    # Saham dianggap "diakumulasi senyap" jika:
    #   CMF 60d > threshold DAN StdDev harga 20d < threshold
    "BANDAR_CMF_THRESHOLD":       0.05,   # CMF 60d minimal (positif lemah sudah cukup)
    "BANDAR_TIGHTNESS_MAX_STD":   0.03,   # StdDev % harga max 3% → harga "dijaga"
    "BANDAR_OBV_LOOKBACK":        10,     # Hari lookback untuk slope OBV

    # ── [V3.6] Dividend Detector ──────────────────────────────────────────────
    # Bulan-bulan historis di mana saham BEI biasa membagikan dividen.
    # Digunakan untuk mendeteksi potensi pre-dividend run-up.
    # Format: bulan angka. Bisa dikustomisasi per ticker di DIVIDEND_CALENDAR.
    "DIVIDEND_SEASON_MONTHS":     [4, 5, 6],   # April-Juni (setelah RUPS)
    "DIVIDEND_CMF_SPIKE_MIN":     0.10,          # CMF spike minimal untuk alert
    "DIVIDEND_RUNUP_DAYS":        45,            # Hari look-back untuk run-up
    "DIVIDEND_TRAP_RETURN_MAX":   0.15,          # Return sudah >15% → potensi trap

    # ── [V3.6] Multi-Timeframe ────────────────────────────────────────────────
    "HOURLY_MA_PERIOD":           20,    # MA periode untuk chart hourly
    "HOURLY_LOOKBACK_DAYS":       5,     # Ambil 5 hari data hourly (cukup untuk MA20)

    # ── [V3.6] Sectoral Heatmap ───────────────────────────────────────────────
    # Threshold CMF rata-rata per sektor untuk label "Hot" / "Cold"
    "SECTOR_HOT_CMF":             0.05,
    "SECTOR_COLD_CMF":           -0.05,

    # ── Watchlist dengan Metadata Sektor & Dividen ────────────────────────────
    # Format: {"ticker": "KODE.JK", "sector": "Nama Sektor", "div_months": [bulan]}
    # div_months: bulan historis cum-date dividen (kosongkan [] jika tidak diketahui)
    "STOCK_UNIVERSE": [
        # === PERBANKAN & KEUANGAN ===
        {"ticker": "BBCA.JK", "sector": "Perbankan", "div_months": [4, 9]},
        {"ticker": "BBRI.JK", "sector": "Perbankan", "div_months": [4, 9]},
        {"ticker": "BMRI.JK", "sector": "Perbankan", "div_months": [4, 9]},
        {"ticker": "BBNI.JK", "sector": "Perbankan", "div_months": [4, 9]},
        {"ticker": "BRIS.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "BTPS.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "BTPN.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "MEGA.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "PNBN.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "BDMN.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "BNGA.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "NISP.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "BJBR.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "BJTM.JK", "sector": "Perbankan", "div_months": [5]},
        {"ticker": "ADMF.JK", "sector": "Keuangan",  "div_months": [5]},
        {"ticker": "BFIN.JK", "sector": "Keuangan",  "div_months": [5]},
        {"ticker": "MFIN.JK", "sector": "Keuangan",  "div_months": [5]},
        # === ENERGI & BATU BARA ===
        {"ticker": "ADRO.JK", "sector": "Energi",    "div_months": [4, 9]},
        {"ticker": "PTBA.JK", "sector": "Energi",    "div_months": [6]},
        {"ticker": "ITMG.JK", "sector": "Energi",    "div_months": [4]},
        {"ticker": "HRUM.JK", "sector": "Energi",    "div_months": [5]},
        {"ticker": "BUMI.JK", "sector": "Energi",    "div_months": []},
        {"ticker": "GEMS.JK", "sector": "Energi",    "div_months": [4]},
        {"ticker": "BSSR.JK", "sector": "Energi",    "div_months": [4]},
        {"ticker": "MEDC.JK", "sector": "Energi",    "div_months": [6]},
        {"ticker": "PGAS.JK", "sector": "Energi",    "div_months": [6]},
        {"ticker": "ELSA.JK", "sector": "Energi",    "div_months": [6]},
        {"ticker": "AKRA.JK", "sector": "Energi",    "div_months": [6]},
        # === LOGAM & TAMBANG ===
        {"ticker": "ANTM.JK", "sector": "Tambang",   "div_months": [5]},
        {"ticker": "INCO.JK", "sector": "Tambang",   "div_months": [6]},
        {"ticker": "MDKA.JK", "sector": "Tambang",   "div_months": []},
        {"ticker": "TINS.JK", "sector": "Tambang",   "div_months": [5]},
        {"ticker": "AMMN.JK", "sector": "Tambang",   "div_months": []},
        {"ticker": "NCKL.JK", "sector": "Tambang",   "div_months": []},
        {"ticker": "MBMA.JK", "sector": "Tambang",   "div_months": []},
        # === PROPERTI ===
        {"ticker": "BSDE.JK", "sector": "Properti",  "div_months": [6]},
        {"ticker": "CTRA.JK", "sector": "Properti",  "div_months": [6]},
        {"ticker": "SMRA.JK", "sector": "Properti",  "div_months": [6]},
        {"ticker": "PWON.JK", "sector": "Properti",  "div_months": [6]},
        {"ticker": "ASRI.JK", "sector": "Properti",  "div_months": [6]},
        {"ticker": "DMAS.JK", "sector": "Properti",  "div_months": [5]},
        {"ticker": "SSIA.JK", "sector": "Properti",  "div_months": [5]},
        # === KONSTRUKSI & INFRASTRUKTUR ===
        {"ticker": "JSMR.JK", "sector": "Infrastruktur", "div_months": [5]},
        {"ticker": "WIKA.JK", "sector": "Infrastruktur", "div_months": []},
        {"ticker": "PTPP.JK", "sector": "Infrastruktur", "div_months": []},
        {"ticker": "WSKT.JK", "sector": "Infrastruktur", "div_months": []},
        {"ticker": "TOWR.JK", "sector": "Infrastruktur", "div_months": [4]},
        {"ticker": "TBIG.JK", "sector": "Infrastruktur", "div_months": [4]},
        {"ticker": "MTEL.JK", "sector": "Infrastruktur", "div_months": [4]},
        # === TEKNOLOGI & TELEKOMUNIKASI ===
        {"ticker": "TLKM.JK", "sector": "Teknologi", "div_months": [4, 9]},
        {"ticker": "ISAT.JK", "sector": "Teknologi", "div_months": [5]},
        {"ticker": "EXCL.JK", "sector": "Teknologi", "div_months": [5]},
        {"ticker": "GOTO.JK", "sector": "Teknologi", "div_months": []},
        {"ticker": "BUKA.JK", "sector": "Teknologi", "div_months": []},
        {"ticker": "EMTK.JK", "sector": "Teknologi", "div_months": [5]},
        {"ticker": "SCMA.JK", "sector": "Teknologi", "div_months": [5]},
        {"ticker": "MNCN.JK", "sector": "Teknologi", "div_months": [5]},
        # === KONSUMER & RETAIL ===
        {"ticker": "INDF.JK", "sector": "Konsumer",  "div_months": [5]},
        {"ticker": "ICBP.JK", "sector": "Konsumer",  "div_months": [5]},
        {"ticker": "MYOR.JK", "sector": "Konsumer",  "div_months": [5]},
        {"ticker": "UNVR.JK", "sector": "Konsumer",  "div_months": [4, 7]},
        {"ticker": "KLBF.JK", "sector": "Konsumer",  "div_months": [6]},
        {"ticker": "SIDO.JK", "sector": "Konsumer",  "div_months": [4]},
        {"ticker": "AMRT.JK", "sector": "Konsumer",  "div_months": [5]},
        {"ticker": "MIDI.JK", "sector": "Konsumer",  "div_months": [5]},
        {"ticker": "MAPI.JK", "sector": "Konsumer",  "div_months": [5]},
        {"ticker": "ACES.JK", "sector": "Konsumer",  "div_months": [5]},
        {"ticker": "GGRM.JK", "sector": "Konsumer",  "div_months": [5]},
        {"ticker": "HMSP.JK", "sector": "Konsumer",  "div_months": [4, 9]},
        {"ticker": "WIIM.JK", "sector": "Konsumer",  "div_months": [5]},
        # === AGRIKULTUR ===
        {"ticker": "CPIN.JK", "sector": "Agrikultur", "div_months": [5]},
        {"ticker": "JPFA.JK", "sector": "Agrikultur", "div_months": [5]},
        {"ticker": "AALI.JK", "sector": "Agrikultur", "div_months": [5]},
        {"ticker": "LSIP.JK", "sector": "Agrikultur", "div_months": [5]},
        {"ticker": "SSMS.JK", "sector": "Agrikultur", "div_months": [5]},
        {"ticker": "TBLA.JK", "sector": "Agrikultur", "div_months": [5]},
        {"ticker": "DSNG.JK", "sector": "Agrikultur", "div_months": [5]},
        {"ticker": "SIMP.JK", "sector": "Agrikultur", "div_months": [5]},
        # === INDUSTRI DASAR & OTOMOTIF ===
        {"ticker": "ASII.JK", "sector": "Industri",  "div_months": [5]},
        {"ticker": "AUTO.JK", "sector": "Industri",  "div_months": [5]},
        {"ticker": "SMSM.JK", "sector": "Industri",  "div_months": [5]},
        {"ticker": "SMGR.JK", "sector": "Industri",  "div_months": [5]},
        {"ticker": "INTP.JK", "sector": "Industri",  "div_months": [5]},
        {"ticker": "TKIM.JK", "sector": "Industri",  "div_months": [5]},
        {"ticker": "INKP.JK", "sector": "Industri",  "div_months": [5]},
        {"ticker": "BRPT.JK", "sector": "Industri",  "div_months": [5]},
        {"ticker": "TPIA.JK", "sector": "Industri",  "div_months": [5]},
        # === HEALTHCARE ===
        {"ticker": "MIKA.JK", "sector": "Healthcare", "div_months": [5]},
        {"ticker": "SILO.JK", "sector": "Healthcare", "div_months": [5]},
        {"ticker": "HEAL.JK", "sector": "Healthcare", "div_months": [5]},
        {"ticker": "KLBF.JK", "sector": "Healthcare", "div_months": [6]},
        {"ticker": "KAEF.JK", "sector": "Healthcare", "div_months": [5]},
        {"ticker": "TSPC.JK", "sector": "Healthcare", "div_months": [5]},
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# SETUP LOGGING
# Format scannable: timestamp + level + message
# Di GitHub Actions, semua output ke stdout akan tersimpan di run log.
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

def get_stock_data(
    ticker: str,
    period_days: int = 90,
    interval: str = "1d",
) -> pd.DataFrame | None:
    """
    Mengambil data OHLCV dari Yahoo Finance.

    Args:
        ticker      : Kode saham format yfinance ("BBCA.JK")
        period_days : Rentang hari yang diminta
        interval    : "1d" untuk daily, "1h" untuk hourly

    Real-Time Fix: period=f"{period_days}d" menghitung mundur dari sekarang,
    menghindari timezone mismatch WIB/UTC yang membuat data tertinggal 1 hari.

    Threshold: period_days * 0.5 — fleksibel terhadap libur bursa.
    BEI tutup ~10 hari/bulan → 30 hari kalender ≈ 15 baris aktual.
    """
    try:
        df = yf.download(
            ticker,
            period=f"{period_days}d",
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if df.empty or len(df) < (period_days * 0.5):
            logger.warning(f"[{ticker}] Data tidak cukup: {len(df)} baris ({interval}).")
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
        logger.error(f"[{ticker}] Gagal ambil data ({interval}): {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 2: INDIKATOR TEKNIKAL (Pure Pandas / NumPy)
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI — Wilder's Smoothed Moving Average."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calc_macd(
    close: pd.Series,
    fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """MACD Line dan Signal Line."""
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _calc_bollinger_bands(
    close: pd.Series, period: int = 20, n_std: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """Bollinger Bands Upper dan Lower."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return sma + (n_std * std), sma - (n_std * std)


def _calc_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K dan %D."""
    lowest_low   = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    price_range  = (highest_high - lowest_low).replace(0, np.nan)
    pct_k = ((close - lowest_low) / price_range) * 100
    pct_d = pct_k.rolling(window=d_period).mean()
    return pct_k, pct_d


def _calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV — On-Balance Volume kumulatif."""
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()


def _calc_cmf(
    high: pd.Series, low: pd.Series,
    close: pd.Series, volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    CMF — Chaikin Money Flow.
    Range: -1.0 hingga +1.0
    Formula: Σ(MFM × Vol, period) / Σ(Vol, period)
    MFM = [(Close − Low) − (High − Close)] / (High − Low)
    """
    hl_range = (high - low).replace(0, np.nan)
    mfm      = ((close - low) - (high - close)) / hl_range
    mfv      = mfm * volume
    return mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()


def _calc_obv_slope(obv: pd.Series, lookback: int = 10) -> float:
    """
    Menghitung slope (kemiringan) OBV dalam N hari terakhir.
    Positif → OBV sedang naik (volume masuk lebih banyak dari keluar).
    Menggunakan linear regression sederhana (polyfit degree 1).
    """
    recent = obv.dropna().tail(lookback)
    if len(recent) < 2:
        return 0.0
    x = np.arange(len(recent))
    try:
        slope = float(np.polyfit(x, recent.values, 1)[0])
        return slope
    except Exception:
        return 0.0


# ── Fitur ML ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Close", "Volume",
    "MA5", "MA20", "MA_Signal", "Price_Change",
    "RSI14", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower",
    "Stoch_K", "Stoch_D",
    "OBV", "CMF",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung semua fitur teknikal untuk ML.

    [V3.1 FIX PRESERVED] Tidak ada dropna() di sini.
    Baris terakhir Target = NaN (intentional).
    Pemisahan live_data / df_train dilakukan di train_and_predict().
    """
    df = df.copy()

    df["MA5"]          = df["Close"].rolling(window=5).mean()
    df["MA20"]         = df["Close"].rolling(window=20).mean()
    df["MA_Signal"]    = df["MA5"] - df["MA20"]
    df["Price_Change"] = df["Close"].pct_change() * 100

    df["RSI14"]                   = _calc_rsi(df["Close"], period=14)
    df["MACD"], df["MACD_Signal"] = _calc_macd(df["Close"])
    df["Stoch_K"], df["Stoch_D"]  = _calc_stochastic(
        df["High"], df["Low"], df["Close"],
        k_period=CONFIG["STOCH_K"], d_period=CONFIG["STOCH_D"],
    )
    df["BB_Upper"], df["BB_Lower"] = _calc_bollinger_bands(
        df["Close"], period=CONFIG["BB_PERIOD"], n_std=CONFIG["BB_STD"],
    )
    df["OBV"] = _calc_obv(df["Close"], df["Volume"])
    df["CMF"] = _calc_cmf(
        df["High"], df["Low"], df["Close"], df["Volume"],
        period=CONFIG["CMF_PERIOD"],
    )

    # Target binary: 1 = besok naik, 0 = besok turun
    future_close = df["Close"].shift(-1)
    df["Target"] = np.where(future_close > df["Close"], 1, 0).astype(float)
    df.iloc[-1, df.columns.get_loc("Target")] = np.nan  # Baris terakhir: NaN

    return df


def generate_ai_analysis(last_row: pd.Series) -> str:
    """Rule-based AI Analysis string dari kondisi teknikal terkini."""
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
        if rsi >= 70:
            parts.append(f"RSI OB ({rsi_i}) ⚠️")
        elif rsi <= 30:
            parts.append(f"RSI OS ({rsi_i}) 🔥")
        else:
            parts.append(f"RSI Netral ({rsi_i})")

    cmf = safe("CMF")
    if pd.notna(cmf):
        if cmf > 0.1:
            parts.append("Akumulasi 🐳")
        elif cmf < -0.1:
            parts.append("Distribusi ⚠️")
        else:
            parts.append("Vol Netral")

    stoch_k = safe("Stoch_K")
    if pd.notna(stoch_k):
        if stoch_k >= 80:
            parts.append("Stoch OB")
        elif stoch_k <= 20:
            parts.append("Stoch OS 🔥")

    return ", ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 3: STOCK SCREENER
# ══════════════════════════════════════════════════════════════════════════════

def screen_stocks(universe: list[dict]) -> list[dict]:
    """
    Filter awal berdasarkan harga, volume, tren, dan status suspend.

    Filter yang diterapkan:
        1. Harga per lembar dalam range CONFIG
        2. Volume rata-rata 20 hari ≥ MIN_AVG_VOLUME
        3. Anti-Falling Knife: Close > MA20
        4. Anti-Suspend: volume 2 hari terakhir > 0 DAN StdDev harga 5 hari > 0

    Anti-rate-limit: time.sleep(0.5) per ticker.
    """
    logger.info("=" * 60)
    logger.info(f"[SCREENER] Memulai screening {len(universe)} saham...")
    logger.info("=" * 60)

    passed = []

    for item in universe:
        ticker  = item["ticker"]
        sector  = item.get("sector", "Unknown")
        div_months = item.get("div_months", [])

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
        if not (min_p <= current_price <= max_p):
            continue
        if avg_volume < CONFIG["MIN_AVG_VOLUME"]:
            continue

        # Filter 3: Anti-Falling Knife
        if current_price <= ma20_current:
            logger.info(f"  ✗ {ticker} DOWNTREND: Close ≤ MA20")
            continue

        # Filter 4: Anti-Suspend / Saham Tidur
        recent_volume  = float(df["Volume"].tail(2).sum())
        recent_std     = float(df["Close"].tail(5).std())
        if recent_volume == 0 or recent_std == 0.0:
            logger.info(f"  ✗ {ticker} SUSPEND: volume mati atau harga stagnan")
            continue

        logger.info(
            f"  ✓ LOLOS | {ticker} | Sektor: {sector} | "
            f"Rp{current_price:,.0f} | Vol {avg_volume:,.0f}"
        )
        passed.append({
            "ticker":        ticker,
            "sector":        sector,
            "div_months":    div_months,
            "current_price": current_price,
            "price_per_lot": price_per_lot,
            "avg_volume":    avg_volume,
        })

    logger.info(f"\n[SCREENER] Selesai: {len(passed)}/{len(universe)} lolos.\n")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 4: ML TRACK — TRIBRID CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def run_ml_track(ticker: str) -> dict | None:
    """
    [TRACK 1] Tribrid Classifier dengan Majority Vote.

    Pipeline:
        1. Ambil data LOOKBACK_DAYS + 30 hari buffer
        2. create_features() → 15 fitur teknikal
        3. Pisah: live_data (hari ini) & df_train (historis bersih)
        4. StandardScaler + train 3 classifier
        5. Majority Vote: ≥ 2/3 model vote Naik
        6. Win Probability: rata-rata predict_proba(kelas=1)

    Kenapa Precision bukan Accuracy?
        Precision mengukur: "dari semua sinyal beli, berapa % yang benar?"
        Lebih relevan untuk trading di mana False Positive langsung merugikan.

    Returns:
        Dict hasil ML, atau None jika tidak lolos.
    """
    df_raw = get_stock_data(ticker, period_days=CONFIG["LOOKBACK_DAYS"] + 30)
    if df_raw is None:
        return None

    df_full = create_features(df_raw)
    missing = [c for c in FEATURE_COLS if c not in df_full.columns]
    if missing:
        return None

    # [V3.1 FIX] Pisahkan live vs training SEBELUM dropna
    live_data = df_full.iloc[-1:]
    df_train  = df_full.iloc[:-1][FEATURE_COLS + ["Target"]].dropna()

    if len(df_train) < 20:
        return None

    X = df_train[FEATURE_COLS].values
    y = df_train["Target"].values.astype(int)

    if len(np.unique(y)) < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["TEST_SIZE"], shuffle=False,
    )
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr  = LogisticRegression(max_iter=CONFIG["LR_MAX_ITER"], random_state=CONFIG["LR_RANDOM_STATE"])
    rf  = RandomForestClassifier(n_estimators=CONFIG["RF_N_ESTIMATORS"], random_state=CONFIG["RF_RANDOM_STATE"], n_jobs=-1)
    xgb = XGBClassifier(
        n_estimators=CONFIG["XGB_N_ESTIMATORS"], learning_rate=CONFIG["XGB_LEARNING_RATE"],
        max_depth=CONFIG["XGB_MAX_DEPTH"], random_state=CONFIG["XGB_RANDOM_STATE"],
        tree_method="hist", verbosity=0, eval_metric="logloss",
    )

    lr.fit(X_train_sc, y_train)
    rf.fit(X_train_sc, y_train)
    xgb.fit(X_train_sc, y_train)

    # Evaluasi ensemble (majority vote) pada test set
    ensemble_test = (
        (lr.predict(X_test_sc) + rf.predict(X_test_sc) + xgb.predict(X_test_sc)) >= 2
    ).astype(int)
    precision = precision_score(y_test, ensemble_test, zero_division=0)
    accuracy  = accuracy_score(y_test, ensemble_test)

    # Prediksi pada data live (hari ini)
    live_feats = live_data[FEATURE_COLS].fillna(0).values
    last_sc    = scaler.transform(live_feats)

    lr_pred  = int(lr.predict(last_sc)[0])
    rf_pred  = int(rf.predict(last_sc)[0])
    xgb_pred = int(xgb.predict(last_sc)[0])
    agree_count = lr_pred + rf_pred + xgb_pred

    lr_prob  = float(lr.predict_proba(last_sc)[0][1])
    rf_prob  = float(rf.predict_proba(last_sc)[0][1])
    xgb_prob = float(xgb.predict_proba(last_sc)[0][1])
    win_prob = (lr_prob + rf_prob + xgb_prob) / 3.0

    passed_ml = (agree_count >= 2) and (win_prob >= CONFIG["MIN_WIN_PROBABILITY"])

    logger.info(
        f"  [ML] {ticker} | Vote: {agree_count}/3 | "
        f"WinProb: {win_prob:.1%} | Precision: {precision:.1%} | "
        f"{'✓ LOLOS' if passed_ml else '✗ GAGAL'}"
    )

    return {
        "passed_ml":     passed_ml,
        "agree_count":   agree_count,
        "win_prob":      win_prob,
        "lr_prob":       lr_prob,
        "rf_prob":       rf_prob,
        "xgb_prob":      xgb_prob,
        "precision":     precision,
        "accuracy":      accuracy,
        "live_last_row": live_data.iloc[0],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 5: BANDAR TRACK  [V3.6 NEW]
# ══════════════════════════════════════════════════════════════════════════════

def run_bandar_track(ticker: str) -> dict:
    """
    [TRACK 2] Deteksi akumulasi institusional (Bandarmologi).

    Tiga komponen:

    1. CMF 60-Hari (Long-Term Accumulation):
       Chaikin Money Flow dengan window 60 hari menangkap akumulasi
       SENYAP yang tidak terlihat di CMF 20 hari standar.
       Bandar biasanya akumulasi perlahan selama 1-3 bulan sebelum
       menggerakkan harga.

    2. Price Tightness (Harga "Dijaga"):
       StdDev harga penutupan 20 hari < 3% menunjukkan volatilitas
       sangat rendah meski volume masuk — pola klasik bandar yang
       "menjaga" harga di range sempit selama akumulasi.
       Setelah akumulasi selesai, biasanya diikuti breakout tajam.

    3. OBV Slope (Konfirmasi Arah Volume):
       Slope positif dari linear regression OBV 10 hari terakhir
       mengkonfirmasi bahwa volume masuk secara konsisten —
       bukan fluktuasi acak.

    Scoring: 0-3 (satu poin per komponen yang terpenuhi)

    Returns:
        Dict berisi nilai indikator Bandar Track dan skor (0-3).
        Selalu return dict (tidak return None) agar Radar Bandar
        tetap bisa menampilkan semua saham yang lolos screener.
    """
    result = {
        "cmf_long":       0.0,
        "price_tightness": 0.0,
        "obv_slope":      0.0,
        "bandar_score":   0,
        "is_accumulating": False,
    }

    # Butuh data lebih panjang untuk CMF 60 hari
    df = get_stock_data(ticker, period_days=CONFIG["CMF_LONG_DAYS"] + 30)
    if df is None:
        return result

    try:
        # 1. CMF 60-Hari
        cmf_long = _calc_cmf(
            df["High"], df["Low"], df["Close"], df["Volume"],
            period=CONFIG["CMF_LONG_DAYS"],
        )
        cmf_long_val = float(cmf_long.dropna().iloc[-1]) if not cmf_long.dropna().empty else 0.0

        # 2. Price Tightness: StdDev % dari harga penutupan 20 hari
        recent_close   = df["Close"].tail(20)
        price_mean     = float(recent_close.mean())
        price_std_pct  = float(recent_close.std() / price_mean) if price_mean > 0 else 1.0

        # 3. OBV Slope
        obv_series = _calc_obv(df["Close"], df["Volume"])
        obv_slope  = _calc_obv_slope(obv_series, lookback=CONFIG["BANDAR_OBV_LOOKBACK"])

        # Scoring
        score = 0
        if cmf_long_val >= CONFIG["BANDAR_CMF_THRESHOLD"]:
            score += 1
        if price_std_pct <= CONFIG["BANDAR_TIGHTNESS_MAX_STD"]:
            score += 1
        if obv_slope > 0:
            score += 1

        is_accumulating = score >= 2  # Butuh minimal 2 dari 3 komponen

        logger.info(
            f"  [BANDAR] {ticker} | CMF60d: {cmf_long_val:+.3f} | "
            f"Tightness: {price_std_pct:.2%} | OBV Slope: {obv_slope:+.0f} | "
            f"Score: {score}/3 | {'🐳 AKUMULASI' if is_accumulating else '—'}"
        )

        return {
            "cmf_long":        cmf_long_val,
            "price_tightness": price_std_pct,
            "obv_slope":       obv_slope,
            "bandar_score":    score,
            "is_accumulating": is_accumulating,
        }

    except Exception as e:
        logger.warning(f"  [BANDAR] {ticker} error: {e}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 6: DIVIDEND DETECTOR  [V3.6 NEW]
# ══════════════════════════════════════════════════════════════════════════════

def run_dividend_detector(ticker: str, div_months: list[int]) -> dict:
    """
    [TRACK 4] Pre-Dividend Run-up Detector & Dividend Trap Filter.

    Logika:
        1. Cek apakah bulan sekarang atau bulan depan termasuk dalam
           div_months (bulan historis cum-date dividen saham ini).
        2. Jika ya, cek apakah ada CMF spike dalam 45 hari terakhir
           (indikasi bandar akumulasi sebelum dividen).
        3. Dividend Trap Filter: hitung return harga 45 hari terakhir.
           Jika sudah naik > 15%, saham mungkin sudah "priced in" dan
           berisiko turun setelah cum-date (sell on the news).

    Pre-Dividend Run-up:
        Pola klasik: bandar masuk 1-2 bulan sebelum cum-date,
        mengangkat harga, lalu jual setelah dividen dibagikan.
        Mendeteksi entry awal ke pola ini memberikan keuntungan ganda:
        capital gain dari run-up + yield dividen itu sendiri.

    Returns:
        Dict berisi status dividen dan alert flags.
    """
    now_month  = datetime.now().month
    next_month = (now_month % 12) + 1

    result = {
        "is_dividend_season":   False,
        "has_cmf_spike":        False,
        "is_dividend_trap":     False,
        "pre_runup_return":     0.0,
        "div_alert":            "",
    }

    if not div_months:
        return result

    # Apakah sekarang musim menjelang dividen?
    is_season = (now_month in div_months) or (next_month in div_months)
    result["is_dividend_season"] = is_season

    if not is_season:
        return result

    df = get_stock_data(ticker, period_days=CONFIG["DIVIDEND_RUNUP_DAYS"] + 10)
    if df is None:
        return result

    try:
        # CMF spike: apakah CMF 20d > threshold dalam periode lookback?
        cmf_series = _calc_cmf(
            df["High"], df["Low"], df["Close"], df["Volume"],
            period=CONFIG["CMF_PERIOD"],
        ).dropna()
        cmf_max_recent = float(cmf_series.tail(20).max()) if not cmf_series.empty else 0.0
        has_spike = cmf_max_recent >= CONFIG["DIVIDEND_CMF_SPIKE_MIN"]
        result["has_cmf_spike"] = has_spike

        # Return harga dalam DIVIDEND_RUNUP_DAYS terakhir
        price_start = float(df["Close"].iloc[0])
        price_now   = float(df["Close"].iloc[-1])
        pre_return  = (price_now - price_start) / price_start if price_start > 0 else 0.0
        result["pre_runup_return"] = pre_return

        # Dividend Trap check
        is_trap = pre_return >= CONFIG["DIVIDEND_TRAP_RETURN_MAX"]
        result["is_dividend_trap"] = is_trap

        # Compose alert string
        if is_trap:
            result["div_alert"] = f"⚠️ Div Trap? Sudah naik {pre_return:.1%} (45d)"
        elif has_spike and is_season:
            result["div_alert"] = f"💰 Pre-Div Run-up! CMF spike +{cmf_max_recent:.2f}"
        elif is_season:
            result["div_alert"] = "📅 Musim Dividen — pantau CMF"

        logger.info(
            f"  [DIV] {ticker} | Season: {is_season} | "
            f"CMF Spike: {has_spike} ({cmf_max_recent:+.2f}) | "
            f"Return: {pre_return:.1%} | Trap: {is_trap}"
        )

    except Exception as e:
        logger.warning(f"  [DIV] {ticker} error: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 7: MULTI-TIMEFRAME CONFIRMATION  [V3.6 NEW]
# ══════════════════════════════════════════════════════════════════════════════

def run_multi_timeframe(ticker: str) -> dict:
    """
    [TRACK 5] Konfirmasi sinyal daily dengan chart hourly.

    Logika:
        Ambil data 1-jam (interval="1h") untuk 5 hari terakhir.
        Hitung MA20 dari data hourly.
        Jika Close hourly terkini > MA20 hourly → tren jangka pendek bullish.

    Kenapa Multi-Timeframe?
        Sinyal daily yang bagus bisa saja muncul di saat tren hourly sedang
        downtrend jangka pendek (misalnya koreksi intraday dalam uptrend besar).
        Konfirmasi hourly memastikan entry timing lebih baik — menghindari
        masuk tepat saat saham sedang tekanan jual jangka pendek.

    Keterbatasan:
        Yahoo Finance membatasi data hourly ke 60 hari terakhir maksimum.
        Untuk 5 hari, biasanya mendapat ~35-40 candle hourly (jam bursa saja).
        Cukup untuk menghitung MA20 hourly.

    Returns:
        Dict berisi status tren hourly dan nilai MA.
    """
    result = {
        "hourly_bullish": False,
        "hourly_close":   0.0,
        "hourly_ma20":    0.0,
        "hourly_error":   False,
    }

    df_h = get_stock_data(
        ticker,
        period_days=CONFIG["HOURLY_LOOKBACK_DAYS"],
        interval="1h",
    )

    if df_h is None or len(df_h) < CONFIG["HOURLY_MA_PERIOD"]:
        result["hourly_error"] = True
        logger.info(f"  [MTF] {ticker} | Data hourly tidak cukup — skip.")
        return result

    try:
        ma20_h     = df_h["Close"].rolling(window=CONFIG["HOURLY_MA_PERIOD"]).mean()
        close_h    = float(df_h["Close"].iloc[-1])
        ma20_h_val = float(ma20_h.dropna().iloc[-1])

        bullish    = close_h > ma20_h_val

        result.update({
            "hourly_bullish": bullish,
            "hourly_close":   close_h,
            "hourly_ma20":    ma20_h_val,
        })

        logger.info(
            f"  [MTF] {ticker} | Hourly Close: {close_h:.0f} | "
            f"Hourly MA20: {ma20_h_val:.0f} | "
            f"{'✓ Bullish' if bullish else '✗ Bearish'}"
        )

    except Exception as e:
        logger.warning(f"  [MTF] {ticker} error: {e}")
        result["hourly_error"] = True

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 8: SECTORAL HEATMAP  [V3.6 NEW]
# ══════════════════════════════════════════════════════════════════════════════

def build_sectoral_heatmap(all_results: list[dict]) -> dict[str, dict]:
    """
    [TRACK 3] Bangun peta panas sektoral berdasarkan rata-rata CMF 60d.

    Mengagregasi nilai cmf_long dari semua saham yang sudah dianalisis
    (termasuk yang tidak lolos ML filter) per sektor, lalu menghitung
    rata-rata untuk mendeteksi rotasi sektor.

    Interpretasi:
        Sektor dengan CMF rata-rata tinggi → institutional inflow ke sektor ini
        Sektor dengan CMF rata-rata rendah → institutional outflow / distribusi

    Contoh output: {"Perbankan": {"avg_cmf": 0.12, "count": 5, "label": "🔥 Hot"}}

    Args:
        all_results : List dict semua saham yang sudah melewati run_bandar_track()

    Returns:
        Dict sektor → statistik CMF
    """
    sector_data: dict[str, list[float]] = {}

    for r in all_results:
        sector  = r.get("sector", "Unknown")
        cmf_val = r.get("bandar", {}).get("cmf_long", None)
        if cmf_val is not None and np.isfinite(cmf_val):
            sector_data.setdefault(sector, []).append(cmf_val)

    heatmap = {}
    for sector, cmf_list in sector_data.items():
        if not cmf_list:
            continue
        avg_cmf = float(np.mean(cmf_list))
        count   = len(cmf_list)

        if avg_cmf >= CONFIG["SECTOR_HOT_CMF"]:
            label = "🔥 Hot (Inflow)"
        elif avg_cmf <= CONFIG["SECTOR_COLD_CMF"]:
            label = "🧊 Cold (Outflow)"
        else:
            label = "➡️  Netral"

        heatmap[sector] = {
            "avg_cmf": avg_cmf,
            "count":   count,
            "label":   label,
        }

    # Urutkan dari CMF tertinggi ke terendah
    heatmap = dict(
        sorted(heatmap.items(), key=lambda x: x[1]["avg_cmf"], reverse=True)
    )

    logger.info(f"[HEATMAP] {len(heatmap)} sektor dianalisis.")
    for s, v in heatmap.items():
        logger.info(f"  {s:<16} CMF avg: {v['avg_cmf']:+.3f} ({v['count']} saham) {v['label']}")

    return heatmap


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 9: ORCHESTRATOR — ANALISIS PER SAHAM
# ══════════════════════════════════════════════════════════════════════════════

def analyze_stock(stock_info: dict) -> dict | None:
    """
    Menjalankan semua track analisis untuk satu saham dan
    menggabungkan hasilnya ke dalam satu dict.

    Track yang dijalankan secara berurutan:
        1. ML Track           → passed_ml, win_prob, precision
        2. Bandar Track       → cmf_long, tightness, obv_slope, score
        3. Dividend Detector  → pre-runup alert, trap filter
        4. Multi-Timeframe    → hourly MA20 confirmation

    Catatan desain:
        Semua track selalu dijalankan (tidak short-circuit).
        Ini memastikan Radar Bandar tetap mendapat data CMF bahkan untuk
        saham yang tidak lolos ML filter, sehingga heatmap dan radar
        tetap komprehensif.

    Returns:
        Dict lengkap semua track, atau None jika data sangat tidak cukup.
    """
    ticker     = stock_info["ticker"]
    sector     = stock_info.get("sector", "Unknown")
    div_months = stock_info.get("div_months", [])

    logger.info(f"\n{'─' * 50}")
    logger.info(f"[ANALISIS] {ticker} ({sector})")

    # ── Track 1: ML ───────────────────────────────────────────────────────────
    ml = run_ml_track(ticker)
    if ml is None:
        logger.info(f"  [ML] {ticker} — data tidak cukup, skip.")
        ml = {
            "passed_ml": False, "agree_count": 0, "win_prob": 0.0,
            "lr_prob": 0.0, "rf_prob": 0.0, "xgb_prob": 0.0,
            "precision": 0.0, "accuracy": 0.0, "live_last_row": pd.Series(dtype=float),
        }

    # ── Track 2: Bandar ───────────────────────────────────────────────────────
    bandar = run_bandar_track(ticker)

    # ── Track 3: Dividend ─────────────────────────────────────────────────────
    dividend = run_dividend_detector(ticker, div_months)

    # ── Track 4: Multi-Timeframe ──────────────────────────────────────────────
    mtf = run_multi_timeframe(ticker)

    # ── AI Analysis string ────────────────────────────────────────────────────
    live_row    = ml.get("live_last_row", pd.Series(dtype=float))
    ai_analysis = generate_ai_analysis(live_row)

    # ── Safe float helper ─────────────────────────────────────────────────────
    def sf(series_or_val, key=None, default=0.0) -> float:
        try:
            val = series_or_val.get(key, np.nan) if key else series_or_val
            result = float(val)
            return result if np.isfinite(result) else default
        except Exception:
            return default

    return {
        # Identitas
        "ticker":        ticker,
        "sector":        sector,
        "div_months":    div_months,
        "current_price": stock_info["current_price"],
        "price_per_lot": stock_info["price_per_lot"],
        # Track 1: ML
        "passed_ml":     ml["passed_ml"],
        "agree_count":   ml["agree_count"],
        "win_prob":      ml["win_prob"],
        "lr_prob":       ml["lr_prob"],
        "rf_prob":       ml["rf_prob"],
        "xgb_prob":      ml["xgb_prob"],
        "precision":     ml["precision"],
        # Track 2: Bandar
        "bandar":        bandar,
        "cmf_long":      bandar["cmf_long"],
        "bandar_score":  bandar["bandar_score"],
        "is_accumulating": bandar["is_accumulating"],
        # Track 3: Dividend
        "dividend":      dividend,
        "div_alert":     dividend["div_alert"],
        "is_div_trap":   dividend["is_dividend_trap"],
        # Track 4: MTF
        "mtf":           mtf,
        "hourly_bullish": mtf["hourly_bullish"],
        # AI Analysis
        "ai_analysis":   ai_analysis,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 10: TELEGRAM DASHBOARD FORMATTER  [V3.6 ROMBAK]
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_obv(obv: float) -> str:
    a = abs(obv)
    if a >= 1_000_000_000:
        return f"{obv / 1_000_000_000:+.2f}M lot"
    elif a >= 1_000_000:
        return f"{obv / 1_000_000:+.1f}Jt"
    return f"{obv:+,.0f}"


def format_dashboard(
    all_results: list[dict],
    heatmap: dict[str, dict],
) -> list[str]:
    """
    Memformat seluruh hasil analisis menjadi Telegram Dashboard 5-seksi.

    Dashboard dibagi menjadi BEBERAPA PESAN (list of str) untuk menghindari
    4096-karakter limit Telegram. Setiap seksi dikirim sebagai pesan terpisah.

    Seksi:
        MSG 1 — Header + Seksi 1: Top ML Signals (Top 10 Win Probability)
        MSG 2 — Seksi 2: Bandar Radar (Top 8 Akumulasi CMF 60d)
        MSG 3 — Seksi 3: Sectoral Heatmap
        MSG 4 — Seksi 4: Dividend Detector Alerts
        MSG 5 — Seksi 5: Distribusi / Guyuran (Bottom 5 CMF)

    Pemisahan per pesan memastikan setiap seksi terbaca jelas dan
    tidak terpotong di tengah-tengah.
    """
    now = datetime.now().strftime("%d %b %Y, %H:%M WIB")
    messages = []

    # Kelompokkan: lolos ML vs semua (untuk radar)
    vip = [r for r in all_results if r.get("passed_ml", False)]

    # ══════════════════════════════════════════════════════════════════
    #  PESAN 1: Header + Seksi 1 — ML Signal Top 10
    # ══════════════════════════════════════════════════════════════════
    lines = [
        "🤖 *SmartAlpha Predictor v3.6*",
        f"📅 {now}   |   🧠 Hybrid Detective",
        f"🔬 ML + Bandar Track + Div Detector + MTF",
        f"📊 Screened: {len(all_results)} | ML Passed: {len(vip)}",
        "━" * 36,
        "",
        "🏆 *SEKSI 1 — ML SIGNAL: TOP 10 WIN PROBABILITY*",
        "_(Majority Vote ≥ 2/3 Model + Prob ≥ 53%)_",
        "",
    ]

    if not vip:
        lines += [
            "⚠️ Tidak ada saham yang lolos filter ML hari ini.",
            "Cek Radar Bandar & Heatmap untuk peluang lain.",
            "",
        ]
    else:
        top_ml = sorted(vip, key=lambda x: x.get("win_prob", 0), reverse=True)[:10]
        for i, r in enumerate(top_ml, 1):
            ticker     = r["ticker"].replace(".JK", "")
            win_prob   = r.get("win_prob", 0.0)
            precision  = r.get("precision", 0.0)
            agree      = r.get("agree_count", 0)
            hourly_ok  = r.get("hourly_bullish", False)
            is_accum   = r.get("is_accumulating", False)
            div_alert  = r.get("div_alert", "")

            # Badge multi-konfirmasi
            badges = []
            if is_accum:
                badges.append("🐳 Akumulasi")
            if hourly_ok:
                badges.append("⏱ MTF ✓")
            if div_alert and "Run-up" in div_alert:
                badges.append("💰 Pre-Div")
            badge_str = "  ".join(badges)

            lines += [
                f"🟢 *{i}. {ticker}* ({r.get('sector', '?')})",
                f"   Harga/Lot  : Rp{r.get('price_per_lot', 0):>9,.0f}",
                f"   Prob. Naik : *{win_prob:.1%}*  ({agree}/3 Models Agree)",
                f"   Precision  : {precision:.1%}",
                f"   🤖 _{r.get('ai_analysis', 'N/A')}_",
            ]
            if badge_str:
                lines.append(f"   🏅 {badge_str}")
            if div_alert:
                lines.append(f"   {div_alert}")
            lines.append("")

    messages.append("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════
    #  PESAN 2: Seksi 2 — Bandar Radar (CMF 60d Akumulasi)
    # ══════════════════════════════════════════════════════════════════
    lines = [
        "🐳 *SEKSI 2 — BANDAR RADAR: TOP 8 AKUMULASI*",
        "_(CMF 60d + Price Tightness + OBV Slope)_",
        "_(Saham di bawah sedang diakumulasi senyap oleh institusi)_",
        "",
    ]

    accum_sorted = sorted(
        all_results,
        key=lambda x: (x.get("bandar_score", 0), x.get("cmf_long", 0)),
        reverse=True,
    )[:8]

    for i, r in enumerate(accum_sorted, 1):
        ticker  = r["ticker"].replace(".JK", "")
        bd      = r.get("bandar", {})
        score   = bd.get("bandar_score", 0)
        cmf60   = bd.get("cmf_long", 0.0)
        tight   = bd.get("price_tightness", 1.0)
        slope   = bd.get("obv_slope", 0.0)
        ml_ok   = r.get("passed_ml", False)
        score_stars = "⭐" * score

        lines += [
            f"  {i}. *{ticker:<6}* ({r.get('sector', '?')})  {score_stars}",
            f"     CMF 60d: `{cmf60:+.3f}`  |  Tightness: `{tight:.2%}`  |  OBV↗: `{slope:+.0f}`",
            f"     ML Signal: {'✅ Lolos' if ml_ok else '❌ Belum lolos'}",
            "",
        ]

    lines += [
        "━" * 36,
        "",
        "⚠️ *SEKSI 2B — DISTRIBUSI: TOP 5 BANDAR KELUAR*",
        "_(CMF 60d paling negatif — tekanan jual institusional)_",
        "",
    ]

    distrib_sorted = sorted(
        all_results, key=lambda x: x.get("cmf_long", 0)
    )[:5]

    for i, r in enumerate(distrib_sorted, 1):
        ticker = r["ticker"].replace(".JK", "")
        cmf60  = r.get("cmf_long", 0.0)
        lines.append(
            f"  {i}. *{ticker:<6}* CMF 60d `{cmf60:+.3f}`  ({r.get('sector', '?')})"
        )

    messages.append("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════
    #  PESAN 3: Seksi 3 — Sectoral Heatmap
    # ══════════════════════════════════════════════════════════════════
    lines = [
        "🗺️ *SEKSI 3 — SECTORAL HEATMAP*",
        "_(Rata-rata CMF 60d per sektor = arah rotasi institusional)_",
        "",
    ]

    if not heatmap:
        lines.append("⚠️ Data heatmap tidak tersedia.")
    else:
        for sector, data in heatmap.items():
            avg  = data["avg_cmf"]
            cnt  = data["count"]
            lbl  = data["label"]
            bar_len = min(int(abs(avg) * 50), 10)
            bar = ("█" * bar_len) if avg >= 0 else ("▒" * bar_len)
            lines.append(
                f"  {lbl}  *{sector:<16}*\n"
                f"     CMF avg: `{avg:+.3f}` ({cnt} saham)  {bar}"
            )
            lines.append("")

    lines += [
        "📌 Sektoral Hot → aliran dana institusi masuk ke sektor ini.",
        "   Fokus screening di sektor Hot untuk peluang terbaik.",
    ]

    messages.append("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════
    #  PESAN 4: Seksi 4 — Dividend Detector
    # ══════════════════════════════════════════════════════════════════
    lines = [
        "💰 *SEKSI 4 — DIVIDEND DETECTOR*",
        "_(Pre-run-up scanner + Dividend Trap filter)_",
        "",
    ]

    div_alerts = [r for r in all_results if r.get("div_alert", "")]
    pre_runups = [r for r in div_alerts if not r.get("is_div_trap", False)]
    traps      = [r for r in div_alerts if r.get("is_div_trap", False)]

    if pre_runups:
        lines += ["🟢 *Pre-Dividend Run-up Opportunities:*", ""]
        for r in sorted(pre_runups, key=lambda x: x.get("cmf_long", 0), reverse=True):
            ticker = r["ticker"].replace(".JK", "")
            div    = r.get("dividend", {})
            ret    = div.get("pre_runup_return", 0.0)
            cmf_sp = div.get("has_cmf_spike", False)
            months = r.get("div_months", [])
            lines += [
                f"  ✓ *{ticker}* ({r.get('sector', '?')})",
                f"    Return 45d: {ret:+.1%}  |  CMF Spike: {'Ya 🐳' if cmf_sp else 'Belum'}",
                f"    Div Month Historis: {months}",
                f"    {r.get('div_alert', '')}",
                "",
            ]
    else:
        lines += ["Tidak ada pre-run-up terdeteksi saat ini.", ""]

    if traps:
        lines += ["🔴 *Potensi Dividend Trap (Sudah Naik Terlalu Tinggi):*", ""]
        for r in traps:
            ticker = r["ticker"].replace(".JK", "")
            ret    = r.get("dividend", {}).get("pre_runup_return", 0.0)
            lines.append(
                f"  ⚠️ *{ticker}* — Sudah naik {ret:+.1%} (45d) — hati-hati!"
            )
        lines.append("")

    messages.append("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════
    #  PESAN 5: Footer + Disclaimer
    # ══════════════════════════════════════════════════════════════════
    lines = [
        "━" * 36,
        "📋 *RINGKASAN EKSEKUTIF*",
        "",
        f"  ML Signals Hari Ini  : {len(vip)} saham",
        f"  Akumulasi Bandar     : {sum(1 for r in all_results if r.get('is_accumulating'))} saham",
        f"  Pre-Div Opportunity  : {len(pre_runups)} saham",
        f"  Potensi Div Trap     : {len(traps)} saham",
        f"  Sektor Terpanas      : {list(heatmap.keys())[0] if heatmap else 'N/A'}",
        "",
        "━" * 36,
        "⚙️ Deployed via GitHub Actions | 16:10 WIB daily",
        "⚠️ *Disclaimer:* Analisis ini untuk tujuan edukasi & riset.",
        "   Bukan rekomendasi beli/jual. Lakukan due diligence sendiri.",
    ]
    messages.append("\n".join(lines))

    return messages


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 11: TELEGRAM SENDER
# ══════════════════════════════════════════════════════════════════════════════

async def send_dashboard(messages: list[str]) -> bool:
    """
    Mengirim list pesan dashboard ke Telegram secara berurutan.

    Setiap pesan dalam list dikirim terpisah dengan jedah 1 detik.
    Jika satu pesan > 4000 karakter, dipotong otomatis per 4000 karakter.
    """
    token   = CONFIG["TELEGRAM_TOKEN"]
    chat_id = CONFIG["TELEGRAM_CHAT_ID"]

    if "YOUR_TELEGRAM_TOKEN_HERE" in token or "YOUR_CHAT_ID_HERE" in chat_id:
        logger.warning(
            "⚠️  Token Telegram belum dikonfigurasi!\n"
            "    Set GitHub Secrets:\n"
            "    TELEGRAM_TOKEN = <token_dari_botfather>\n"
            "    TELEGRAM_CHAT_ID = <chat_id_kamu>"
        )
        logger.info("=== PREVIEW DASHBOARD ===")
        for i, msg in enumerate(messages, 1):
            print(f"\n--- PESAN {i} ---\n{msg}")
        logger.info("=========================")
        return False

    try:
        bot = telegram.Bot(token=token)
        for i, message in enumerate(messages, 1):
            # Auto-chunk jika satu pesan masih > 4000 karakter
            chunks = [message[j:j + 4000] for j in range(0, len(message), 4000)]
            for chunk in chunks:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="Markdown",
                )
                await asyncio.sleep(0.5)
            logger.info(f"  ✓ Pesan {i}/{len(messages)} terkirim.")
            await asyncio.sleep(1.0)

        logger.info(f"✓ Dashboard ({len(messages)} pesan) berhasil terkirim.")
        return True

    except telegram.error.TelegramError as e:
        logger.error(f"Gagal mengirim Telegram: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  MODUL 12: MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

async def run_pipeline():
    """
    Orkestrasi pipeline SmartAlpha v3.6 end-to-end.

    Flow:
        [STOCK_UNIVERSE]
               ↓  time.sleep(0.5) per ticker
        [SCREENER]  harga + volume + anti-falling-knife + anti-suspend
               ↓
        [PER SAHAM — 4 track paralel secara sekuensial]
          ├─ [ML TRACK]       LogReg + RF + XGB → Majority Vote
          ├─ [BANDAR TRACK]   CMF 60d + Tightness + OBV Slope
          ├─ [DIV DETECTOR]   Pre-run-up + Trap Filter
          └─ [MTF]            Hourly MA20 Confirmation
               ↓
        [SECTORAL HEATMAP]  Agregasi CMF per sektor
               ↓
        [TELEGRAM DASHBOARD]  5 seksi × 5 pesan terpisah
    """
    logger.info("=" * 60)
    logger.info("🚀 SmartAlpha Predictor v3.6 — Pipeline dimulai...")
    logger.info(f"   Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}")
    logger.info("=" * 60)
    t0 = time.time()

    universe = CONFIG["STOCK_UNIVERSE"]
    if not universe:
        logger.error("STOCK_UNIVERSE kosong! Tambahkan saham di CONFIG.")
        return

    # ── Fase 1: Screening ─────────────────────────────────────────────────────
    screened = screen_stocks(universe)
    if not screened:
        logger.warning("Tidak ada saham yang lolos screening.")
        return

    # ── Fase 2: Analisis per Saham ────────────────────────────────────────────
    logger.info(f"\n[ANALISIS] Memulai 4-track analysis untuk {len(screened)} saham...")
    all_results = []

    for stock_info in screened:
        result = analyze_stock(stock_info)
        if result:
            all_results.append(result)

    if not all_results:
        logger.warning("Tidak ada hasil analisis yang valid.")
        return

    ml_passed  = sum(1 for r in all_results if r.get("passed_ml"))
    accum_count = sum(1 for r in all_results if r.get("is_accumulating"))
    logger.info(
        f"\n[SUMMARY] {len(all_results)} dianalisis | "
        f"ML: {ml_passed} lolos | Bandar: {accum_count} akumulasi"
    )

    # ── Fase 3: Sectoral Heatmap ──────────────────────────────────────────────
    logger.info("\n[HEATMAP] Membangun Sectoral Heatmap...")
    heatmap = build_sectoral_heatmap(all_results)

    # ── Fase 4: Format & Kirim Dashboard ─────────────────────────────────────
    logger.info("\n[TELEGRAM] Memformat & mengirim dashboard...")
    messages = format_dashboard(all_results, heatmap)
    await send_dashboard(messages)

    elapsed = time.time() - t0
    logger.info(f"\n✅ Pipeline selesai dalam {elapsed:.1f} detik.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Local Run:
        python main.py

    .env setup (buat file '.env' di folder yang sama):
        TELEGRAM_TOKEN=token_dari_botfather
        TELEGRAM_CHAT_ID=chat_id_kamu

    GitHub Actions: Set TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID sebagai
    Repository Secrets (Settings → Secrets → Actions).
    Lihat .github/workflows/smartalpha.yml untuk konfigurasi cron.
    """
    asyncio.run(run_pipeline())
