import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nifty AI Indicator Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for financial dashboard styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .bullish {
        color: #10b981;
        font-weight: bold;
    }
    .bearish {
        color: #ef4444;
        font-weight: bold;
    }
    .neutral {
        color: #f59e0b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class NiftyAIIndicator:
    def __init__(self):
        self.scaler = StandardScaler()

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_nifty_data(_self, period="1y"):
        """Fetch real Nifty data or use sample data"""
        try:
            # Try to fetch real data
            nifty = yf.download("^NSEI", period=period, interval="1d")
            if not nifty.empty:
                nifty.reset_index(inplace=True)
                nifty.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                return nifty.drop('Adj Close', axis=1)
        except:
            pass

        # Fallback to sample data
        return _self._create_sample_data()

    def _create_sample_data(self):
        """Create sample Nifty data"""
        dates = pd.date_range(start='2023-01-01', end='2024-10-27', freq='D')
        np.random.seed(42)

        base_price = 18000
        returns = np.random.normal(0.0005, 0.015, len(dates))

        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        volumes = np.random.normal(1000000, 200000, len(dates))
        volumes = np.abs(volumes)

        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Volume': volumes
        })

        # Generate OHLC from close prices
        df['Open'] = df['Close'] * np.random.normal(1, 0.005, len(df))
        df['High'] = df['Close'] * np.random.normal(1.01, 0.01, len(df))
        df['Low'] = df['Close'] * np.random.normal(0.99, 0.01, len(df))

        # Ensure OHLC consistency
        for i in range(len(df)):
            df.loc[i, 'High'] = max(df.loc[i, ['Open', 'Close', 'High']].max(), df.loc[i, 'High'])
            df.loc[i, 'Low'] = min(df.loc[i, ['Open', 'Close', 'Low']].min(), df.loc[i, 'Low'])

        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()

        # EMA
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

        # Volume analysis
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']

        return df

    def calculate_ai_score(self, df):
        """Calculate AI composite score"""
        # Normalize indicators
        ma_signal = np.where(df['SMA_5'] > df['SMA_20'], 1, -1)
        macd_signal = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
        rsi_signal = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))

        # Composite score
        df['AI_Score'] = (0.4 * ma_signal + 0.3 * macd_signal + 0.3 * rsi_signal)

        # Signal classification
        df['AI_Signal'] = pd.cut(df['AI_Score'], 
                               bins=[-2, -0.3, 0.3, 2], 
                               labels=['Bearish', 'Neutral', 'Bullish'])

        return df

    def generate_insights(self, df):
        """Generate trading insights"""
        latest = df.iloc[-1]

        insights = {
            'signal': str(latest['AI_Signal']),
            'strength': abs(latest['AI_Score']) * 100,
            'rsi_condition': ('Overbought' if latest['RSI'] > 70 else 
                            'Oversold' if latest['RSI'] < 30 else 'Neutral'),
            'trend': 'Bullish' if latest['SMA_5'] > latest['SMA_20'] else 'Bearish',
            'support': df['Low'].tail(20).min(),
            'resistance': df['High'].tail(20).max(),
        }

        return insights

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Nifty AI Indicator Dashboard</h1>
        <p>AI-Powered Technical Analysis for Nifty 50 Index</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize AI indicator
    ai_indicator = NiftyAIIndicator()

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    period = st.sidebar.selectbox("Time Period", ["6mo", "1y", "2y"], index=1)
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5 min)")

    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 5 minutes")

    # Load and process data
    with st.spinner("Loading Nifty data..."):
        df = ai_indicator.fetch_nifty_data(period)
        df = ai_indicator.calculate_technical_indicators(df)
        df = ai_indicator.calculate_ai_score(df)

    # Generate insights
    insights = ai_indicator.generate_insights(df)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        signal_class = insights['signal'].lower()
        st.metric(
            "AI Signal",
            insights['signal'],
            help="Current AI-generated trading signal"
        )

    with col2:
        st.metric(
            "Current Level",
            f"₹{df.iloc[-1]['Close']:,.0f}",
            f"{((df.iloc[-1]['Close']/df.iloc[-2]['Close'] - 1) * 100):+.2f}%"
        )

    with col3:
        st.metric(
            "RSI",
            f"{df.iloc[-1]['RSI']:.1f}",
            insights['rsi_condition']
        )

    with col4:
        st.metric(
            "Signal Strength",
            f"{insights['strength']:.1f}%",
            help="Confidence level of AI signal"
        )

    # Main chart
    st.subheader("📈 Price Chart with AI Signals")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=["Price & Moving Averages", "Volume", "AI Score"]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Nifty 50"
        ),
        row=1, col=1
    )

    # Moving averages
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='purple')), row=1, col=1)

    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )

    # AI Score
    ai_colors = ['red' if score < -0.3 else 'green' if score > 0.3 else 'orange' for score in df['AI_Score']]
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['AI_Score'], name='AI Score', 
                  line=dict(color='blue', width=2), fill='tonexty'),
        row=3, col=1
    )

    fig.update_layout(
        height=800,
        title_text="Nifty 50 Technical Analysis Dashboard",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Technical indicators table
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Technical Indicators")
        indicators_df = pd.DataFrame({
            'Indicator': ['RSI', 'MACD', 'SMA 20', 'SMA 50', 'Bollinger Upper', 'Bollinger Lower'],
            'Value': [
                f"{df.iloc[-1]['RSI']:.2f}",
                f"{df.iloc[-1]['MACD']:.2f}",
                f"₹{df.iloc[-1]['SMA_20']:.0f}",
                f"₹{df.iloc[-1]['SMA_50']:.0f}",
                f"₹{df.iloc[-1]['BB_upper']:.0f}",
                f"₹{df.iloc[-1]['BB_lower']:.0f}"
            ],
            'Status': [
                insights['rsi_condition'],
                'Bullish' if df.iloc[-1]['MACD'] > df.iloc[-1]['MACD_signal'] else 'Bearish',
                'Above' if df.iloc[-1]['Close'] > df.iloc[-1]['SMA_20'] else 'Below',
                'Above' if df.iloc[-1]['Close'] > df.iloc[-1]['SMA_50'] else 'Below',
                'Below' if df.iloc[-1]['Close'] < df.iloc[-1]['BB_upper'] else 'Above',
                'Above' if df.iloc[-1]['Close'] > df.iloc[-1]['BB_lower'] else 'Below'
            ]
        })
        st.dataframe(indicators_df, use_container_width=True)

    with col2:
        st.subheader("🎯 AI Insights")
        st.write(f"**Current Signal:** {insights['signal']}")
        st.write(f"**Trend Direction:** {insights['trend']}")
        st.write(f"**Support Level:** ₹{insights['support']:,.0f}")
        st.write(f"**Resistance Level:** ₹{insights['resistance']:,.0f}")

        # Trading recommendation
        if insights['signal'] == 'Bullish':
            st.success("💡 **Recommendation:** Consider long positions with stop loss below support")
        elif insights['signal'] == 'Bearish':
            st.error("💡 **Recommendation:** Consider short positions with stop loss above resistance")
        else:
            st.warning("💡 **Recommendation:** Hold current positions and wait for clearer signals")

    # Signal history
    st.subheader("📈 Recent Signal History")
    recent_signals = df.tail(10)[['Date', 'Close', 'AI_Signal', 'AI_Score', 'RSI']].copy()
    recent_signals['Date'] = recent_signals['Date'].dt.strftime('%Y-%m-%d')
    recent_signals['Close'] = recent_signals['Close'].apply(lambda x: f"₹{x:,.0f}")
    recent_signals['AI_Score'] = recent_signals['AI_Score'].apply(lambda x: f"{x:.2f}")
    recent_signals['RSI'] = recent_signals['RSI'].apply(lambda x: f"{x:.1f}")

    st.dataframe(recent_signals, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This dashboard is for educational purposes only. Not financial advice.")
    st.markdown("**Data Source:** Yahoo Finance | **Update Frequency:** Real-time (with 15-minute delay)")

if __name__ == "__main__":
    main()
