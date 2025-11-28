# quick_check_env.py
from dotenv import load_dotenv
import os, sys
load_dotenv()
print("Python exec:", sys.executable)
print("PWD:", os.getcwd())
print("MONGO_CONNECTION_STRING preview:", (os.getenv("MONGO_CONNECTION_STRING") or "")[:120])
