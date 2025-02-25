import os
import pandas as pd
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def calculate_indicators(df):
    """Compute technical indicators like RSI, MACD, and Bollinger Bands."""
    df['rsi'] = df['close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / \
                df['close'].diff().abs().rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + df['rsi']))
    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['bollinger_mid'] = df['close'].rolling(window=20).mean()
    df['bollinger_upper'] = df['bollinger_mid'] + (df['close'].rolling(window=20).std() * 2)
    df['bollinger_lower'] = df['bollinger_mid'] - (df['close'].rolling(window=20).std() * 2)
    return df

# Load API Keys from .env
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("BASE_URL")

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

# Define stocks to fetch
tickers = ['AAPL', 'MSFT', 'SPY']

def fetch_historical_data(symbol, timeframe='1Day', limit=1000):
    """Fetches historical market data from Alpaca"""
    try:
        barset = api.get_bars(symbol, timeframe, limit=limit).df
        barset['symbol'] = symbol  # Add symbol for reference
        return calculate_indicators(barset)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Fetch and compile data
dataframes = []
for ticker in tickers:
    df = fetch_historical_data(ticker)
    if df is not None:
        dataframes.append(df)

# Ensure we have data before proceeding
if dataframes:
    market_data = pd.concat(dataframes)

    # ✅ Remove non-numeric columns (like 'symbol')
    if 'symbol' in market_data.columns:
        market_data.drop(columns=['symbol'], inplace=True)

    # ✅ Convert all data to numeric values
    market_data = market_data.apply(pd.to_numeric, errors='coerce')

    # ✅ Fill NaN values with 0
    market_data.fillna(0, inplace=True)

    print("✅ Market Data Retrieved, Shape:", market_data.shape)
    print(market_data.head())  # Debugging: Show first few rows

    # AI Training Setup
    print("🚀 AI Training Triggered. Market Data Shape:", market_data.shape)
    print("🚀 Initializing AI Training Environment...")

    class TradingEnv(gym.Env):
        def __init__(self, data):
            super(TradingEnv, self).__init__()
            self.data = data
            self.current_step = 0
            self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32
            )

        def step(self, action):
            """Executes one step in the environment."""
            self.current_step += 1

            # Placeholder reward (replace with real reward function)
            reward = np.random.rand()

            done = self.current_step >= len(self.data) - 1

            # ✅ Ensure observation is float32
            obs = self.data.iloc[self.current_step].values.astype(np.float32)

            # ✅ Add an empty info dictionary
            info = {}

            return obs, reward, done, False, info  # ✅ Fixed return format

        def reset(self, seed=None, options=None):
            """Reset environment and return initial observation & empty info dictionary."""
            self.current_step = 0
            obs = self.data.iloc[self.current_step].values.astype(np.float32)
            return obs, {}

        def seed(self, seed=None):
            pass  # Gym expects a `seed()` function

    print("🚀 Creating PPO Model...")
    env = make_vec_env(lambda: TradingEnv(market_data), n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1)

    print("🚀 Starting AI Training for 10,000 timesteps...")
    model.learn(total_timesteps=10000, progress_bar=True)

    print("🚀 Saving the trained AI model...")
    model.save("ppo_trading_bot")

    print("✅ AI Model Training Complete! Checking if file was saved...")
    
    if os.path.exists("ppo_trading_bot.zip"):
        print("✅ Model saved successfully!")
    else:
        print("❌ Model file was NOT saved!")

else:
    print("❌ No market data retrieved. AI training aborted.")

