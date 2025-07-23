
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import mplfinance as mpf
import tempfile

st.set_page_config(layout="wide")
st.title("📈 Intraday Reversal Screener – Multi-Universe Edition")
st.markdown("Find reversal setups across S&P 500, NASDAQ 100, Russell 2000, or custom U.S. stocks using technical signals.")

@st.cache_data
def get_ticker_list(universe):
    if universe == "S&P 500":
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return table['Symbol'].tolist()
    elif universe == "NASDAQ 100":
        table = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[3]
        return table['Ticker'].tolist()
    elif universe == "Russell 2000":
        url = "https://stockmarketmba.com/stocksinrussell2000.php"
        table = pd.read_html(url)[0]
        return table['Symbol'].tolist()
    else:
        # fallback: use Yahoo Finance tickers from screener
        return pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents-financials.csv")['Symbol'].tolist()

@st.cache_data
def download_data(ticker):
    df = yf.download(ticker, period='7d', interval='1h', progress=False)
    return df

def get_support_resistance(data, lookback=20):
    highs = data['High'].rolling(lookback).max()
    lows = data['Low'].rolling(lookback).min()
    return highs.iloc[-1], lows.iloc[-1]

def detect_patterns(df):
    patterns = []
    for i in range(1, len(df)):
        o, h, l, c = df.iloc[i][['Open', 'High', 'Low', 'Close']]
        prev_o = df.iloc[i - 1]['Open']
        prev_c = df.iloc[i - 1]['Close']
        body = abs(c - o)
        range_ = h - l

        if body < 0.3 * range_ and (h - max(c, o)) < 0.2 * range_ and (min(c, o) - l) > 0.4 * range_:
            patterns.append("Hammer")
        elif c > prev_o and prev_c > o and (c - o) > (prev_o - prev_c):
            patterns.append("Bullish Engulfing")
        elif abs(c - o) < 0.1 * range_:
            patterns.append("Doji")
        else:
            patterns.append("")
    patterns.insert(0, "")
    return patterns

def plot_chart(df, ticker, support=None, resistance=None):
    df = df.copy()
    df.index.name = 'Date'
    bb = ta.volatility.BollingerBands(close=df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['Pattern'] = detect_patterns(df)

    annotations = [
        dict(x=df.index[i], y=df['Close'].iloc[i], text=df['Pattern'].iloc[i],
             arrowstyle='->', color='orange')
        for i in range(len(df)) if df['Pattern'].iloc[i]
    ]

    apds = [
        mpf.make_addplot(df['bb_high'], color='green'),
        mpf.make_addplot(df['bb_low'], color='red')
    ]

    hlines = []
    if support: hlines.append(support)
    if resistance: hlines.append(resistance)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        mpf.plot(
            df.tail(50),
            type='candle',
            style='yahoo',
            title=f'{ticker} - 1H Candles',
            ylabel='Price',
            volume=True,
            addplot=apds,
            alines=annotations,
            hlines=dict(hlines=hlines, colors=['blue', 'purple'], linestyle='--'),
            savefig=tmpfile.name
        )
        return tmpfile.name

st.sidebar.header("Scanner Settings")
universe = st.sidebar.selectbox("Select Market Universe", ["S&P 500", "NASDAQ 100", "Russell 2000"])
max_tickers = st.sidebar.slider("Number of tickers to scan", min_value=10, max_value=150, value=30, step=10)

tickers = get_ticker_list(universe)[:max_tickers]
results = []

with st.spinner(f"Scanning {universe} tickers..."):
    for ticker in tickers:
        try:
            df = download_data(ticker)
            if df.empty or len(df) < 30:
                continue
            df.dropna(inplace=True)

            df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            bb = ta.volatility.BollingerBands(close=df['Close'])
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            df['avg_volume'] = df['Volume'].rolling(window=10).mean()
            df['Pattern'] = detect_patterns(df)

            last = df.iloc[-1]
            recent_df = df.iloc[-10:]
            volume_spike = last['Volume'] > 1.5 * last['avg_volume']
            oversold_confirm = all(recent_df['rsi'] < 35)
            overbought_confirm = all(recent_df['rsi'] > 65)
            resistance, support = get_support_resistance(df)
            price = last['Close']
            near_support = price < support * 1.02
            near_resistance = price > resistance * 0.98

            signal = None
            if oversold_confirm and price < last['bb_low'] and near_support:
                signal = "Bullish Reversal"
            elif overbought_confirm and price > last['bb_high'] and near_resistance:
                signal = "Bearish Reversal"

            pattern_signal = df['Pattern'].iloc[-1]

            if (signal or pattern_signal) and volume_spike:
                results.append({
                    "Ticker": ticker,
                    "Price": round(price, 2),
                    "RSI": round(last['rsi'], 2),
                    "ATR": round(last['atr'], 2),
                    "Volume Spike": "Yes",
                    "Signal": signal or "Pattern",
                    "Pattern": pattern_signal,
                    "Support": round(support, 2),
                    "Resistance": round(resistance, 2),
                    "df": df
                })
        except:
            continue

if results:
    results_df = pd.DataFrame(results).drop(columns=['df'])
    st.success(f"{len(results)} reversal candidates found.")
    st.dataframe(results_df)

    for stock in results:
        with st.expander(f"📊 {stock['Ticker']} - {stock['Signal']}"):
            chart = plot_chart(stock['df'], stock['Ticker'], stock['Support'], stock['Resistance'])
            st.image(chart, use_column_width=True)
else:
    st.warning("No reversal setups found.")
