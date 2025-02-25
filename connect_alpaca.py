from dotenv import load_dotenv
import os
import alpaca_trade_api as tradeapi

# Load API keys from .env
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("BASE_URL")

print("Loaded API Key:", ALPACA_API_KEY)
print("Loaded Secret Key:", ALPACA_SECRET_KEY)
print("Using Base URL:", BASE_URL)

# Initialize Alpaca API connection
try:
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version="v2")
    account = api.get_account()

    print("\n✅ Connection Successful!")
    print("Account ID:", account.id)
    print("Equity:", account.equity)
    print("Buying Power:", account.buying_power)
    print("Status:", account.status)

except Exception as e:
    print("\n❌ ERROR OCCURRED:")
    print(e)

