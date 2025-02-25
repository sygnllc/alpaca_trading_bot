from dotenv import load_dotenv
import os

load_dotenv()

print("API Key:", os.getenv("ALPACA_API_KEY"))
print("Secret Key:", os.getenv("ALPACA_SECRET_KEY"))

