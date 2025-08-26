# Importiere benötigte Module.
import pandas as pd  # Für Datenmanipulation und -analyse.
import numpy as np  # Für numerische Berechnungen.
import tkinter as tk  # Für die Erstellung grafischer Benutzeroberflächen (GUI).
from tkinter import messagebox  # Für die Anzeige von Pop-up-Nachrichten.
from PIL import Image, ImageTk  # Für das Laden und Anzeigen von Bildern in der GUI.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Zum Einbetten von Matplotlib-Grafiken in tkinter.
import matplotlib.pyplot as plt  # Zum Erstellen von Diagrammen und Grafiken.
import requests  # Zum Senden von HTTP-Anfragen (z. B. an APIs).
from datetime import datetime  # Für die Arbeit mit Datums- und Zeitangaben.
import os  # Für die Interaktion mit dem Betriebssystem (z. B. Datei- und Pfadoperationen).
from sklearn.ensemble import RandomForestRegressor  # Für maschinelles Lernen (Random Forest Regression).
from sklearn.model_selection import train_test_split  # Zum Aufteilen von Datensätzen in Trainings- und Testdaten.
from sklearn.metrics import mean_squared_error  # Zur Bewertung von Vorhersagemodellen.
from tkinter import font  # Für Schriftarten-Anpassungen in tkinter.
from tkinter import ttk  # Für erweiterte tkinter-Widgets.
import threading  # Für die parallele Ausführung von Aufgaben.
import time  # Für zeitbezogene Funktionen.
from io import BytesIO  # Für die Verarbeitung von Binärdaten im Speicher.
import math # Mathe-Tools (pi, e, Wurzeln, Logarithmen, Sinus, Fakultät usw.).

# Projekt: ZenithAlpha 1.2 von Lukas Völzing. Datum: August 2025. Unternehmen: Linoz Developments.

print()
print("Welcome to ZenithAlpha from Linoz Developments. Version 1.2. Developer: Lukas Voelzing")

# Funktion zum Abrufen von historischen Aktienkursdaten von Alpha Vantage.

def fetch_data_from_alpha_vantage(ticker, api_key):
    try:
        
        # API-URL für tägliche Zeitreihendaten.
        
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        
        # Prüft, ob die gewünschten Daten im JSON enthalten sind.
        
        if 'Time Series (Daily)' not in data:
            print()
            print(f"No historical data available for: {ticker}.")
            return None
        
        # Konvertiert die JSON-Daten in einen pandas DataFrame.
        
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Benennt die Spalten um und sortiert den Index.
        
        df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        print()
        print(f"Historical data retrieved for: {ticker}")
        return df
    except Exception as e:
        print()
        print(f"Error retrieving data from Alpha Vantage: {str(e)}")
        return None
    
# Füge eine sichere Konvertierungsfunktion hinzu.
        
def safe_float(x):
    """Konvertiert robust nach float; bei None/leer -> NaN."""
    try:
        if x is None:
            return math.nan
        if isinstance(x, str) and x.strip().lower() in {"", "none", "nan", "null", "-"}:
            return math.nan
        return float(x)
    except Exception:
        return math.nan

# Funktion zur Bewertung fundamentaler Aktienkennzahlen.

def evaluate_stock_fundamentals_av(ticker, api_key):
    """
    Holt Alpha Vantage OVERVIEW und berechnet eine robuste, fehlertolerante Investment-Bewertung.
    Gibt zurück:
      - Kern-Kennzahlen
      - Score (0–100)
      - Empfehlung (BUY/HOLD/SELL)
      - Tags (Value/Quality/Income/Growth)
      - Kurzbegründung
    """
    def to_float(x):
        
        # Hilfsfunktion zur sicheren Konvertierung in float.
        
        try:
            if x is None or x == "" or str(x).lower() in {"none", "nan", "null", "-"}:
                return None
            return float(x)
        except Exception:
            return None
    try:
        
        # API-Abfrage für fundamentale Daten.
        
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
        resp = requests.get(url, timeout=20)
        info = resp.json()
        
        # Prüft, ob die Antwort gültige Daten enthält.
        
        if not isinstance(info, dict) or "Symbol" not in info:
            print(f"\nNo fundamental data available for: {ticker}. Raw: {info}")
            return None
        
        # --- Extrahiere Rohwerte aus OVERVIEW (fehlerrobust) ---
        
        pe   = to_float(info.get("PERatio"))
        ps   = to_float(info.get("PriceToSalesRatioTTM") or info.get("PriceToSalesRatio"))
        pb   = to_float(info.get("PriceToBookRatio"))
        peg  = to_float(info.get("PEGRatio"))
        div_yield = to_float(info.get("DividendYield"))          # z. B. 0.02 = 2%
        payout    = to_float(info.get("PayoutRatio"))
        pmargin   = to_float(info.get("ProfitMargin"))           # z. B. 0.12 = 12%
        roe_ttm   = to_float(info.get("ReturnOnEquityTTM"))      # z. B. 0.18 = 18%
        roa_ttm   = to_float(info.get("ReturnOnAssetsTTM"))
        q_rev_yoy = to_float(info.get("QuarterlyRevenueGrowthYOY"))
        q_eps_yoy = to_float(info.get("QuarterlyEarningsGrowthYOY"))
        beta      = to_float(info.get("Beta"))
        cur_ratio = to_float(info.get("CurrentRatio"))           # Optional; nicht immer vorhanden.
        dte       = to_float(info.get("DebtToEquity")) or to_float(info.get("QuarterlyDebtToEquityRatio"))
        mcap      = to_float(info.get("MarketCapitalization"))
        target    = to_float(info.get("AnalystTargetPrice"))
        price     = to_float(info.get("50DayMovingAverage")) or to_float(info.get("200DayMovingAverage"))
        
        # --- Scoring-Logik (fehlertolerant, konservative Schwellen) ---
        
        score = 0
        reasons_pos = []
        reasons_neg = []
        
        # Value-Kriterien.
        
        if pe is not None:
            if pe < 15: score += 25; reasons_pos.append(f"attractive P/E ({pe:.1f})")
            elif pe < 25: score += 15
            elif pe < 40: score += 5
            else: reasons_neg.append(f"high P/E ({pe:.1f})")
        if ps is not None:
            if ps < 2: score += 20; reasons_pos.append(f"low P/S ({ps:.2f})")
            elif ps < 4: score += 10
            else: reasons_neg.append(f"high P/S ({ps:.2f})")
        if pb is not None:
            if pb < 3: score += 10; reasons_pos.append(f"reasonable P/B ({pb:.2f})")
            elif pb < 6: score += 5
            else: reasons_neg.append(f"high P/B ({pb:.2f})")
        if peg is not None and peg > 0:
            if peg < 1: score += 10; reasons_pos.append(f"PEG < 1 ({peg:.2f})")
            elif peg < 2: score += 5
            
        # Quality-Kriterien.
        
        if roe_ttm is not None:
            if roe_ttm > 0.15: score += 12; reasons_pos.append(f"strong ROE ({roe_ttm*100:.1f}%)")
            elif roe_ttm > 0.10: score += 8
            elif roe_ttm > 0.05: score += 4
        if pmargin is not None:
            if pmargin > 0.10: score += 8; reasons_pos.append(f"solid net margin ({pmargin*100:.1f}%)")
            elif pmargin > 0.05: score += 4
            elif pmargin < 0: reasons_neg.append("negative net margin")
            
        # Income-Kriterien.
        
        if div_yield is not None and div_yield > 0:
            if div_yield >= 0.03: score += 10; reasons_pos.append(f"dividend yield {(div_yield*100):.1f}%")
            elif div_yield >= 0.015: score += 5
            if payout is not None and payout > 0.9:
                reasons_neg.append(f"high payout ratio ({payout*100:.0f}%)")
                
        # Growth-Kriterien.
        
        growth_points = 0
        if q_rev_yoy is not None:
            if q_rev_yoy > 0.15: growth_points += 6
            elif q_rev_yoy > 0.05: growth_points += 3
            elif q_rev_yoy < 0: reasons_neg.append("revenue shrinking YoY")
        if q_eps_yoy is not None:
            if q_eps_yoy > 0.15: growth_points += 6
            elif q_eps_yoy > 0.05: growth_points += 3
            elif q_eps_yoy < 0: reasons_neg.append("EPS shrinking YoY")
        score += min(growth_points, 12)
        
        # Risiko/Finanzen.
        
        if beta is not None:
            if 0.8 <= beta <= 1.2: score += 5; reasons_pos.append(f"beta {beta:.2f} (market-like)")
            elif beta > 1.5: score -= 5; reasons_neg.append(f"high volatility (beta {beta:.2f})")
        if cur_ratio is not None and cur_ratio < 1:
            score -= 5; reasons_neg.append(f"current ratio {cur_ratio:.2f} (<1)")
        if dte is not None:
            if dte > 2: score -= 8; reasons_neg.append(f"high debt/equity ({dte:.2f})")
            elif dte < 0.8: score += 4; reasons_pos.append(f"moderate leverage (D/E {dte:.2f})")
            
        # Analysten-Target als weicher Indikator.
        
        upside = None
        if price and target and price > 0:
            upside = (target / price) - 1.0
            if upside > 0.15: score += 4; reasons_pos.append(f"analyst upside {upside*100:.0f}%")
            elif upside < -0.10: score -= 4; reasons_neg.append("analysts see downside")
            
        # Begrenzung des Scores.
        
        score = max(0, min(100, score))
        
        # Empfehlung basierend auf dem Score.
        
        if score >= 70:
            recommendation = "BUY"
        elif score >= 55:
            recommendation = "HOLD (Bias: BUY)"
        elif score >= 40:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"
            
        # Red-Flag-Downgrades.
        
        if pmargin is not None and pmargin < 0 and (div_yield is None or div_yield == 0):
            
            # Verlust + keine Dividende → höchstens HOLD.
            
            if recommendation == "BUY":
                recommendation = "HOLD"
        if dte is not None and dte > 3:
            if recommendation == "BUY":
                recommendation = "HOLD"
                
        # Tags basierend auf den Kriterien.
        
        tags = []
        if pe is not None and ps is not None and pe < 20 and ps < 2.5: tags.append("Value")
        if roe_ttm and pmargin and roe_ttm > 0.12 and pmargin > 0.10: tags.append("Quality")
        if div_yield and div_yield >= 0.02: tags.append("Income")
        if (q_rev_yoy and q_rev_yoy > 0.10) or (q_eps_yoy and q_eps_yoy > 0.10): tags.append("Growth")
        if not tags: tags.append("Neutral")
        
        # Kurzbegründung.
        
        pos_txt = ", ".join(reasons_pos[:4]) if reasons_pos else "solid base"
        neg_txt = ", ".join(reasons_neg[:3]) if reasons_neg else "no significant risks"
        explanation = f"Pro: {pos_txt}. Con: {neg_txt}."
        result = {
            "Symbol": info.get("Symbol"),
            "Name": info.get("Name"),
            "Sector": info.get("Sector"),
            "MarketCap": mcap,
            "PERatio": pe,
            "PriceToSales": ps,
            "PriceToBook": pb,
            "PEGRatio": peg,
            "DividendYield": div_yield,
            "PayoutRatio": payout,
            "ProfitMargin": pmargin,
            "ROE_TTM": roe_ttm,
            "ROA_TTM": roa_ttm,
            "RevGrowth_YoY": q_rev_yoy,
            "EPSGrowth_YoY": q_eps_yoy,
            "Beta": beta,
            "CurrentRatio": cur_ratio,
            "DebtToEquity": dte,
            "AnalystTargetPrice": target,
            "RefPrice": price,
            "Score": score,
            "Tags": tags,
            "recommendation": recommendation,
            "explanation": explanation,
        }
        return result
    except Exception as e:
        print(f"Error in fundamental analysis: {str(e)}")
        return None

# Technische Indikatoren.

def calculate_sma(data, window=50):
    
    # Berechnet den einfachen gleitenden Durchschnitt (SMA).
    
    data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
    return data

def calculate_rsi(data, window=14):
    
    # Berechnet den Relative Strength Index (RSI).
    
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_macd(data):
    
    # Berechnet den MACD-Indikator.
    
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def calculate_bollinger_bands(data, window=20):
    
    # Berechnet Bollinger-Bänder.
    
    data['Bollinger_Middle'] = data['Close'].rolling(window=window).mean()
    data['Bollinger_Upper'] = data['Bollinger_Middle'] + (data['Close'].rolling(window=window).std() * 2)
    data['Bollinger_Lower'] = data['Bollinger_Middle'] - (data['Close'].rolling(window=window).std() * 2)
    return data

def backtest_strategy(data):
    
    # Backtest einer einfachen Handelsstrategie.
    
    data['Signal'] = np.where(
        (data['RSI'] < 30) & (data['MACD'] > data['MACD_Signal']) &
        (data['Close'] > data['SMA_50']) & (data['Close'] < data['Bollinger_Lower']),
        1, 0
    )
    data['Returns'] = data['Close'].pct_change() * data['Signal'].shift(1)
    data['Portfolio_Value'] = (1 + data['Returns']).cumprod() * 1000
    return data

# Maschinelles Lernen.

def train_ml_model(data):
    
    # Trainiert ein RandomForest-Modell zur Vorhersage von Aktienrenditen.
    
    data['Return'] = data['Close'].pct_change().shift(-1)
    if 'SMA_50' not in data.columns:
        data = calculate_sma(data)
    if 'RSI' not in data.columns:
        data = calculate_rsi(data)
    if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:
        data = calculate_macd(data)
    if 'Bollinger_Upper' not in data.columns or 'Bollinger_Lower' not in data.columns:
        data = calculate_bollinger_bands(data)
    data = data.dropna().copy()
    features = ['Close', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower']
    X = data[features]
    y = data['Return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print()
    print(f"ML Model Mean Squared Error: {mse}")
    data['ML_Prediction'] = model.predict(X[features])
    return data, model

def backtest_quant_strategy(data):
    
    # Backtest einer quantitativen Handelsstrategie.
    
    ml_threshold = 0.001
    data['Quant_Signal'] = np.where(
        (data['ML_Prediction'] > ml_threshold) |
        ((data['RSI'] < 30) & (data['MACD'] > data['MACD_Signal']) &
         (data['Close'] > data['SMA_50']) & (data['Close'] < data['Bollinger_Lower'])),
        1, 0
    )
    data['Quant_Returns'] = data['Close'].pct_change() * data['Quant_Signal'].shift(1)
    data['Quant_Portfolio'] = (1 + data['Quant_Returns']).cumprod() * 1000
    return data

def fetch_and_display_gold_chart():
    
    # Lädt und zeigt den Goldpreis-Chart an.
    
    try:
        url = "//charts.gold.de/b/goldkurs_24stunden_usd.jpg"
        response = requests.get(f"https:{url}")
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        img = img.resize((1000, 700), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        chart_window = tk.Toplevel(root)
        chart_window.title("Gold Chart (USD)")
        width, height = 1000, 700
        screen_width = chart_window.winfo_screenwidth()
        screen_height = chart_window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        chart_window.geometry(f"{width}x{height}+{x}+{y}")
        chart_label = tk.Label(chart_window, image=img_tk)
        chart_label.image = img_tk
        chart_label.pack(pady=10)
    except Exception as e:
        messagebox.showerror("Error", f"Error loading Gold chart: {e}")

def fetch_and_display_silver_chart():
    
    # Lädt und zeigt den Silberpreis-Chart an.
    
    try:
        url = "//charts.gold.de/b/silberkurs_24stunden_usd.jpg"
        response = requests.get(f"https:{url}")
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        width, height = 1000, 700
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        chart_window = tk.Toplevel(root)
        chart_window.title("Silver Chart (USD)")
        screen_width = chart_window.winfo_screenwidth()
        screen_height = chart_window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        chart_window.geometry(f"{width}x{height}+{x}+{y}")
        chart_window.resizable(False, False)
        chart_label = tk.Label(chart_window, image=img_tk)
        chart_label.image = img_tk
        chart_label.pack(pady=10)
    except Exception as e:
        messagebox.showerror("Error", f"Error loading Silver chart: {e}")

def show_imprint():
    
    # Zeigt das Impressum an.
    
    imprint_text = (
        "ZenithAlpha - Quantitative Investment Tool 1.2.\n"
        "Developer: Lukas Völzing\n"
        "Company: Linoz Developments\n\n"
        "Contact: lukas.vlzg@outlook.com\n\n"
        "© 2025 All rights reserved."
    )
    imprint_window = tk.Toplevel(root)
    imprint_window.title("Imprint | ZenithAlpha - Quantitative Investment Tool for Hedgefunds and Investors")
    imprint_window.configure(bg="#f0f0f0")
    width = 800
    height = 400
    screen_width = imprint_window.winfo_screenwidth()
    screen_height = imprint_window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    imprint_window.geometry(f"{width}x{height}+{x}+{y}")
    frame = tk.Frame(imprint_window, bg="#f0f0f0")
    frame.pack(expand=True, fill=tk.BOTH)
    label = tk.Label(
        frame,
        text=imprint_text,
        font=("Helvetica", 14),
        bg="#f0f0f0",
        fg="#333333",
        justify=tk.CENTER,
        anchor="center"
    )
    label.pack(expand=True, pady=30, padx=20)

# GUI Funktionen.

def analyze_and_plot():
    
    # Führt die technische Analyse und den Backtest durch.
    
    ticker = entry_ticker.get().upper()
    if not ticker:
        messagebox.showerror("Error", "Please enter a valid ticker symbol!")
        return
    api_key = "XXXXXXXYYYYYYYZZZZ"
    data = fetch_data_from_alpha_vantage(ticker, api_key)
    if data is not None:
        data = calculate_sma(data)
        data = calculate_rsi(data)
        data = calculate_macd(data)
        data = calculate_bollinger_bands(data)
        data = backtest_strategy(data)
        plot_data(data, title="Technical Analysis & Backtest")
        display_investment_opportunities(data)
    else:
        messagebox.showerror("Error", "Error retrieving the data.")

def analyze_fundamentals():
    # Führt die fundamentale Analyse durch.
    ticker = entry_ticker.get().upper()
    if not ticker:
        messagebox.showerror("Error", "Please enter a valid ticker symbol!")
        return

    api_key = "XXXXXXXYYYYYYYZZZZ"
    result = evaluate_stock_fundamentals_av(ticker, api_key)
    if result is None:
        messagebox.showerror("Error", "No fundamental data available.")
        return

    try:
        stock_name = result.get("Name") or ticker

        # WICHTIG: sichere Konvertierung + korrekte Keys
        pe        = safe_float(result.get("PERatio"))
        pb        = safe_float(result.get("PriceToBook") or result.get("PriceToBookRatio"))
        div_yield = safe_float(result.get("DividendYield"))

        score = 0

        # P/E
        if math.isnan(pe):
            pe_text = "P/E: n/a"
        elif pe < 15:
            score += 1
            pe_text = f"P/E: {pe:.2f} (cheap)"
        elif pe > 25:
            score -= 1
            pe_text = f"P/E: {pe:.2f} (expensive)"
        else:
            pe_text = f"P/E: {pe:.2f} (fair)"

        # P/B
        if math.isnan(pb):
            pb_text = "P/B: n/a"
        elif pb < 1:
            score += 1
            pb_text = f"P/B: {pb:.2f} (undervalued)"
        elif pb > 3:
            score -= 1
            pb_text = f"P/B: {pb:.2f} (overvalued)"
        else:
            pb_text = f"P/B: {pb:.2f} (fair)"

        # Dividend Yield
        if math.isnan(div_yield):
            div_text = "Dividend Yield: n/a"
        elif div_yield > 0.03:
            score += 1
            div_text = f"Dividend Yield: {div_yield:.2%} (attractive)"
        else:
            div_text = f"Dividend Yield: {div_yield:.2%} (low)"

        # Verdict
        if score >= 2:
            verdict = "Undervalued! ✅"
            verdict_color = "green"
        elif score <= -1:
            verdict = "Overvalued! ❌"
            verdict_color = "red"
        else:
            verdict = "Fairly Valued! ⚖️"
            verdict_color = "orange"

        # UI
        result_window = tk.Toplevel(root)
        result_window.title(f"Fundamental Analysis: {ticker}")
        width, height = 500, 350
        screen_width = result_window.winfo_screenwidth()
        screen_height = result_window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        result_window.geometry(f"{width}x{height}+{x}+{y}")
        result_window.resizable(False, False)

        tk.Label(result_window, text=f"Analysis for {stock_name} ({ticker})", font=("Arial", 16, "bold")).pack(pady=15)
        tk.Label(result_window, text=pe_text, font=("Arial", 12)).pack(pady=5)
        tk.Label(result_window, text=pb_text, font=("Arial", 12)).pack(pady=5)
        tk.Label(result_window, text=div_text, font=("Arial", 12)).pack(pady=5)
        tk.Label(result_window, text=verdict, font=("Arial", 16, "bold"), fg=verdict_color).pack(pady=20)

    except Exception as e:
        messagebox.showerror("Error", f"Error analyzing fundamentals: {e}")

def analyze_quant_strategy():
    
    # Führt die quantitative Analyse durch.
    
    ticker = entry_ticker.get().upper()
    if not ticker:
        messagebox.showerror("Error", "Please enter a valid ticker symbol!")
        return
    api_key = "XXXXXXXYYYYYYYZZZZ"
    data = fetch_data_from_alpha_vantage(ticker, api_key)
    if data is not None:
        data = calculate_sma(data)
        data = calculate_rsi(data)
        data = calculate_macd(data)
        data = calculate_bollinger_bands(data)
        data, model = train_ml_model(data)
        data = backtest_quant_strategy(data)
        plot_data(data, title="Quantitative Strategy (ML & Technical)")
        recommendation, last_pred = classify_stock_opinion(data)
        pred_percent = round(last_pred * 100, 2)
        messagebox.showinfo(
            "Quantitative Analysis",
            f"Result for {ticker}:\n"
            f"Prediction percent: {pred_percent:+.2f}%\n"
            f"Recommendation: {recommendation}"
        )
        display_quant_investment_opportunities(data)
    else:
        messagebox.showerror("Error", "Error retrieving the data.")

def plot_data(data, title="Analysis"):
    
    # Plottet die Daten in einem Matplotlib-Diagramm.
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Closing price', color='blue')
    ax.yaxis.set_major_formatter('${x:1.2f}')
    if 'SMA_50' in data.columns:
        ax.plot(data.index, data['SMA_50'], label='50-day SMA', color='red')
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        ax.plot(data.index, data['MACD'], label='MACD', color='green')
        ax.plot(data.index, data['MACD_Signal'], label='MACD Signal', color='orange')
    if 'Bollinger_Upper' in data.columns and 'Bollinger_Lower' in data.columns:
        ax.plot(data.index, data['Bollinger_Upper'], label='Bollinger Upper', color='purple', linestyle='--')
        ax.plot(data.index, data['Bollinger_Lower'], label='Bollinger Lower', color='purple', linestyle='--')
    if 'ML_Prediction' in data.columns:
        ax.plot(
            data.index,
            data['ML_Prediction'] * data['Close'].shift(1) + data['Close'].shift(1),
            label='ML Prediction (Price)', color='green', linestyle='--'
        )
    ax.set_title(title)
    ax.legend()
    for i, price in enumerate(data['Close']):
        if i % 10 == 0:
            ax.text(data.index[i], price, f'${price:.2f}', color='blue', fontsize=8)
    global canvas_plot
    try:
        canvas_plot.get_tk_widget().destroy()
    except NameError:
        pass
    canvas_plot = FigureCanvasTkAgg(fig, master=root)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack(pady=10)

def display_investment_opportunities(data):
    
    # Zeigt eine Meldung zu technischen Kaufgelegenheiten an.
    
    last_signal = data['Signal'].iloc[-1]
    status = "Optimal buying opportunity!" if last_signal == 1 else "Currently no buying opportunity."
    messagebox.showinfo("Investment Opportunities (Technical)", status)

def display_quant_investment_opportunities(data):
    
    # Zeigt eine Meldung zu quantitativen Kaufgelegenheiten an.
    
    last_signal = data['Quant_Signal'].iloc[-1]
    status = "Quantitative Buy Signal!" if last_signal == 1 else "No Quantitative Buy Signal."
    messagebox.showinfo("Investment Opportunities (Quantitative)", status)

def classify_stock_opinion(data):
    
    # Klassifiziert die Aktie basierend auf der ML-Vorhersage.
    
    last_pred = data['ML_Prediction'].iloc[-1]
    threshold_buy = 0.002
    threshold_sell = -0.002
    if last_pred > threshold_buy:
        recommendation = "BUY"
    elif last_pred < threshold_sell:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    return recommendation, last_pred

# GUI-Erstellung mit allen Funktionen.

def create_gui():
    global root, entry_ticker, canvas_plot
    root = tk.Tk()
    root.title("Home: ZenithAlpha - Quantitative Investment Tool for Hedgefunds and Investors")
    root.configure(bg="#f0f0f0")
    default_font = ("Helvetica", 12)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
    welcome_label = tk.Label(root, text="Welcome to ZenithAlpha from Linoz Developments. Version 1.2. Developer: Lukas Völzing", font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#000000")
    welcome_label.pack(pady=20)
    logo_frame = tk.Frame(root, bg="#f0f0f0")
    logo_frame.pack(pady=10)
    logo_path_1 = "C:\\Studium Fachhochschule Frankfurt\\X) Private Projekte\\2) Python\\1) Zenith Alpha\\Z) Logo.png"
    if os.path.exists(logo_path_1):
        try:
            logo_image_1 = Image.open(logo_path_1)
            logo_image_1 = logo_image_1.resize((330, 330), Image.Resampling.LANCZOS)
            logo_photo_1 = ImageTk.PhotoImage(logo_image_1)
            logo_label_1 = tk.Label(logo_frame, image=logo_photo_1, bg="#f0f0f0")
            logo_label_1.image = logo_photo_1
            logo_label_1.pack(side=tk.LEFT, padx=10)
        except Exception as e:
            print(f"Error loading the first logo: {e}")
    logo_path_2 = "C:\\Studium Fachhochschule Frankfurt\\X) Private Projekte\\2) Python\\1) Zenith Alpha\\Z) Linoz Developments.png"
    if os.path.exists(logo_path_2):
        try:
            logo_image_2 = Image.open(logo_path_2)
            logo_image_2 = logo_image_2.resize((330, 260), Image.Resampling.LANCZOS)
            logo_photo_2 = ImageTk.PhotoImage(logo_image_2)
            logo_label_2 = tk.Label(logo_frame, image=logo_photo_2, bg="#f0f0f0")
            logo_label_2.image = logo_photo_2
            logo_label_2.pack(side=tk.LEFT, padx=10)
        except Exception as e:
            print(f"Error loading the second logo: {e}")
    input_frame = tk.Frame(root, bg="#f0f0f0")
    input_frame.pack(pady=20)
    label_ticker = tk.Label(input_frame, text="Please enter stock ticker:", font=default_font, bg="#ddd9d9")
    label_ticker.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    entry_ticker = tk.Entry(input_frame, font=default_font, width=30)
    entry_ticker.grid(row=0, column=1, padx=5, pady=5)
    btn_analyze = tk.Button(input_frame, text="Technical Analysis & Backtest", command=analyze_and_plot, font=default_font, bg="#4CAF50", fg="white")
    btn_analyze.grid(row=1, column=0, padx=5, pady=10, sticky=tk.E+tk.W)
    btn_fundamental = tk.Button(input_frame, text="Fundamental Analysis", command=analyze_fundamentals, font=default_font, bg="#2196F3", fg="white")
    btn_fundamental.grid(row=1, column=1, padx=5, pady=10, sticky=tk.E+tk.W)
    btn_quant = tk.Button(input_frame, text="Quantitative Strategy (ML + Technical)", command=analyze_quant_strategy, font=default_font, bg="#E68D09", fg="white")
    btn_quant.grid(row=1, column=2, padx=5, pady=10, sticky=tk.E+tk.W)
    btn_gold_chart = tk.Button(input_frame, text="Show Gold Price Chart", command=fetch_and_display_gold_chart, font=default_font, bg="#FFD700", fg="black")
    btn_gold_chart.grid(row=4, column=0, columnspan=3, sticky=tk.E+tk.W, padx=5, pady=10)
    btn_silver_chart = tk.Button(input_frame, text="Show Silver Chart", command=fetch_and_display_silver_chart, font=default_font, bg="#C0C0C0", fg="black")
    btn_silver_chart.grid(row=7, column=0, columnspan=3, sticky=tk.E+tk.W, padx=5, pady=10)
    btn_imprint = tk.Button(input_frame, text="Imprint", command=show_imprint, font=default_font, bg="#2C2828", fg="white")
    btn_imprint.grid(row=3, column=0, columnspan=4, sticky=tk.E + tk.W, padx=6, pady=10)
    root.mainloop()

# Hauptprogramm.

if __name__ == "__main__":
    create_gui()
