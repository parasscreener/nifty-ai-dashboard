# First, let me create the core AI indicator logic and data processing functions
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set up basic configuration
np.random.seed(42)

# Create a comprehensive AI-based Nifty indicator class
class NiftyAIIndicator:
    def __init__(self):
        self.features = []
        self.model_weights = {}
        self.indicators_history = []
        self.sentiment_scores = []
        
    def calculate_technical_indicators(self, df):
        """Calculate various technical indicators"""
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Price_momentum_5'] = df['Close'].pct_change(5)
        df['Price_momentum_10'] = df['Close'].pct_change(10)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        return df
    
    def create_sample_nifty_data(self, days=500):
        """Create sample Nifty data for demonstration"""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Generate realistic Nifty-like price movements
        base_price = 18000
        returns = np.random.normal(0.0005, 0.015, days)  # Daily returns with slight positive bias
        
        # Add some market regime changes
        regime_change_points = [100, 200, 350]
        for point in regime_change_points:
            if point < days:
                returns[point:point+50] += np.random.normal(0.01, 0.02, min(50, days-point))
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        volumes = np.random.normal(1000000, 200000, days)
        volumes = np.abs(volumes)  # Ensure positive volumes
        
        # Generate OHLC data
        df = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.normal(1, 0.005) for p in prices],
            'High': [p * np.random.normal(1.01, 0.01) for p in prices],
            'Low': [p * np.random.normal(0.99, 0.01) for p in prices],
            'Close': prices,
            'Volume': volumes
        })
        
        # Ensure OHLC consistency
        for i in range(len(df)):
            high = max(df.loc[i, 'Open'], df.loc[i, 'Close']) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(df.loc[i, 'Open'], df.loc[i, 'Close']) * (1 - abs(np.random.normal(0, 0.005)))
            df.loc[i, 'High'] = max(df.loc[i, 'High'], high)
            df.loc[i, 'Low'] = min(df.loc[i, 'Low'], low)
        
        return df
    
    def apply_ai_pattern_recognition(self, df):
        """Apply AI-based pattern recognition"""
        patterns = []
        
        # Moving average crossover patterns
        df['MA_crossover_signal'] = np.where(df['SMA_5'] > df['SMA_20'], 1, 
                                           np.where(df['SMA_5'] < df['SMA_20'], -1, 0))
        
        # MACD pattern recognition
        df['MACD_bullish'] = ((df['MACD'] > df['MACD_signal']) & 
                             (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
        df['MACD_bearish'] = ((df['MACD'] < df['MACD_signal']) & 
                             (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))).astype(int)
        
        # RSI pattern recognition
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
        
        # Bollinger Band pattern
        df['BB_squeeze'] = ((df['Close'] < df['BB_lower']) | (df['Close'] > df['BB_upper'])).astype(int)
        
        return df
    
    def calculate_ai_composite_score(self, df):
        """Calculate AI composite score using weighted indicators"""
        # Define weights for different indicators
        weights = {
            'ma_trend': 0.25,
            'macd': 0.20,
            'rsi': 0.15,
            'volume': 0.15,
            'momentum': 0.15,
            'volatility': 0.10
        }
        
        # Normalize indicators to -1 to 1 range
        ma_score = np.tanh(df['MA_crossover_signal'] * 2)
        
        macd_score = np.tanh((df['MACD_bullish'] - df['MACD_bearish']) * 2)
        
        rsi_score = np.tanh((50 - df['RSI']) / 20)  # Inverted so low RSI = positive signal
        
        volume_score = np.tanh((df['Volume_ratio'] - 1) * 2)
        
        momentum_score = np.tanh(df['Price_momentum_5'] * 50)
        
        volatility_score = -np.tanh((df['Volatility'] / df['Close']) * 100)  # Lower volatility = positive
        
        # Calculate composite score
        df['AI_Score'] = (weights['ma_trend'] * ma_score +
                         weights['macd'] * macd_score +
                         weights['rsi'] * rsi_score +
                         weights['volume'] * volume_score +
                         weights['momentum'] * momentum_score +
                         weights['volatility'] * volatility_score)
        
        # Generate signal categories
        df['AI_Signal'] = pd.cut(df['AI_Score'], 
                               bins=[-2, -0.3, 0.3, 2], 
                               labels=['Bearish', 'Neutral', 'Bullish'])
        
        return df
    
    def generate_market_insights(self, df):
        """Generate AI-driven market insights"""
        latest = df.iloc[-1]
        recent = df.tail(20)
        
        insights = {
            'current_signal': str(latest['AI_Signal']),
            'signal_strength': float(latest['AI_Score']),
            'trend_direction': 'Upward' if latest['SMA_5'] > latest['SMA_20'] else 'Downward',
            'rsi_condition': ('Overbought' if latest['RSI'] > 70 else 
                            'Oversold' if latest['RSI'] < 30 else 'Neutral'),
            'volume_analysis': ('Above Average' if latest['Volume_ratio'] > 1.2 else
                              'Below Average' if latest['Volume_ratio'] < 0.8 else 'Normal'),
            'volatility_level': ('High' if latest['Volatility'] > recent['Volatility'].quantile(0.8) else
                               'Low' if latest['Volatility'] < recent['Volatility'].quantile(0.2) else 'Medium'),
            'support_level': float(recent['Low'].min()),
            'resistance_level': float(recent['High'].max()),
            'confidence_score': min(100, max(0, 50 + (abs(latest['AI_Score']) * 25)))
        }
        
        return insights

# Initialize and test the indicator
nifty_ai = NiftyAIIndicator()

# Create sample data
print("Creating sample Nifty data...")
df = nifty_ai.create_sample_nifty_data(300)

# Calculate technical indicators
print("Calculating technical indicators...")
df = nifty_ai.calculate_technical_indicators(df)

# Apply AI pattern recognition
print("Applying AI pattern recognition...")
df = nifty_ai.apply_ai_pattern_recognition(df)

# Calculate composite AI score
print("Calculating AI composite score...")
df = nifty_ai.calculate_ai_composite_score(df)

# Generate insights
print("Generating market insights...")
insights = nifty_ai.generate_market_insights(df)

print("Sample data created successfully!")
print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nLatest insights: {insights}")

# Save sample data for dashboard
df.to_csv('nifty_ai_data.csv', index=False)
print("Sample data saved as 'nifty_ai_data.csv'")

# Display first few rows with key indicators
display_cols = ['Date', 'Close', 'RSI', 'MACD', 'AI_Score', 'AI_Signal']
print(f"\nSample of processed data:")
print(df[display_cols].tail(10))