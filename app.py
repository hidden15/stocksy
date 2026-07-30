import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian AI Stock Predictor & Portfolio Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Indian Market AI Predictor & Analyzer")
st.markdown("Integrates Machine Learning predictions with Portfolio Optimization tailored for Indian investors.")

# --- Auto-Format Indian Tickers ---
def format_in_ticker(symbol):
    symbol = symbol.strip().upper()
    # Automatically append .NS (NSE) if the user forgets it
    if symbol and not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        return f"{symbol}.NS"
    return symbol

# --- Sidebar Inputs ---
st.sidebar.header("User Settings")

# Investor Risk Profile
investor_profile = st.sidebar.selectbox(
    "Select Risk Profile",
    ["Conservative (Low Risk)", "Moderate (Balanced)", "Aggressive (High Risk)"]
)

# Indian Stock Selection
raw_ticker = st.sidebar.text_input("Enter NSE Stock Ticker (e.g., IRFC, RELIANCE, ZOMATO)", "IRFC")
ticker = format_in_ticker(raw_ticker)

raw_portfolio = st.sidebar.text_input(
    "Enter Basket Tickers for Portfolio (comma separated)", 
    "RELIANCE, TCS, HDFCBANK, INFY"
)
portfolio_tickers = [format_in_ticker(t) for t in raw_portfolio.split(",") if t.strip()]

time_period = st.sidebar.selectbox("Historical Data Horizon", ["2y", "5y", "10y"], index=1)

# --- Data Caching Functions ---
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period):
    data = yf.download(symbol, period=period)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data.reset_index(inplace=True)
    return data

@st.cache_data(ttl=3600)
def fetch_portfolio_data(tickers, period):
    data = yf.download(tickers, period=period)['Close']
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data.dropna()

# --- Helper Functions ---
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Target: 1 if next day close is higher than current close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

def train_ml_model(df):
    features = ['SMA_20', 'SMA_50', 'RSI', 'Volume']
    X = df[features]
    y = df['Target']
    
    # Train-test split (chronological for time series)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Predict next day movement
    latest_features = X.iloc[[-1]]
    next_day_pred = model.predict(latest_features)[0]
    next_day_prob = model.predict_proba(latest_features)[0][next_day_pred]
    
    return accuracy, next_day_pred, next_day_prob

# --- Main App Execution ---
tabs = st.tabs(["📊 Stock Predictor", "💼 Portfolio Suggester"])

# TAB 1: Stock Prediction
with tabs[0]:
    st.subheader(f"Single Stock Analysis: {ticker}")
    df_raw = fetch_stock_data(ticker, time_period)
    
    if not df_raw.empty:
        df = add_technical_indicators(df_raw.copy())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price"))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name="20-Day SMA"))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="50-Day SMA"))
            fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### AI Movement Prediction")
            acc, pred, prob = train_ml_model(df)
            
            direction = "🟢 UP" if pred == 1 else "🔴 DOWN"
            st.metric(label="Model Direction Prediction (Next Session)", value=direction)
            st.write(f"**Confidence Level:** {prob * 100:.2f}%")
            st.write(f"**Historical Model Accuracy:** {acc * 100:.2f}%")
            
            st.divider()
            st.markdown("### Technical Indicators")
            latest_rsi = df['RSI'].iloc[-1]
            st.metric(label="Current RSI (14)", value=f"{latest_rsi:.2f}")
            if latest_rsi > 70:
                st.warning("Status: Overbought Area (High Risk to Buy)")
            elif latest_rsi < 30:
                st.info("Status: Oversold Area (Potential Buying Opportunity)")
            else:
                st.write("Status: Neutral Zone")
    else:
        st.error(f"Unable to load data for {ticker}. Ensure the stock is listed on NSE/BSE.")

# TAB 2: Portfolio Optimization
with tabs[1]:
    st.subheader("Automated Portfolio Optimization (Indian Equities)")
    st.write(f"**Target Profile:** {investor_profile}")
    
    if len(portfolio_tickers) >= 2:
        prices = fetch_portfolio_data(portfolio_tickers, time_period)
        
        if not prices.empty:
            mu = expected_returns.mean_historical_return(prices)
            S = risk_models.sample_cov(prices)
            
            ef = EfficientFrontier(mu, S)
            
            if "Conservative" in investor_profile:
                weights = ef.min_volatility()
                st.info("Optimized for **Minimum Volatility** (Capital Preservation Focus).")
            elif "Aggressive" in investor_profile:
                weights = ef.max_sharpe()
                st.info("Optimized for **Maximum Sharpe Ratio** (High Return Focus).")
            else: # Moderate
                weights = ef.max_sharpe()
                st.info("Optimized for **Balanced Growth** (Targeted Risk-Reward Ratio).")
                
            cleaned_weights = ef.clean_weights()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### Optimal Asset Weights")
                weights_df = pd.DataFrame(list(cleaned_weights.items()), columns=["Ticker", "Allocation Weight"])
                weights_df['Allocation Weight'] = weights_df['Allocation Weight'].apply(lambda x: f"{x*100:.2f}%")
                st.table(weights_df)
                
            with col_b:
                st.markdown("### Allocation Breakdown")
                fig_pie = go.Figure(data=[go.Pie(labels=list(cleaned_weights.keys()), values=list(cleaned_weights.values()), hole=.3)])
                st.plotly_chart(fig_pie, use_container_width=True)
                
            # Portfolio Metrics
            st.divider()
            st.markdown("### Expected Portfolio Performance")
            ret, vol, sharpe = ef.portfolio_performance(verbose=False)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Annual Return", f"{ret*100:.2f}%")
            m2.metric("Annual Volatility (Risk)", f"{vol*100:.2f}%")
            m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        else:
            st.error("Error retrieving historical data for basket items.")
    else:
        st.warning("Please provide at least 2 stock tickers separated by commas for portfolio optimization.")
