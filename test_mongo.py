# # test_mongo.py
# import os
# from dotenv import load_dotenv
# from pymongo import MongoClient
# from pymongo.server_api import ServerApi
# import traceback

# load_dotenv()

# conn = os.getenv("MONGO_CONNECTION_STRING")
# print("MONGO_CONNECTION_STRING (first 120 chars, redacted):", (conn or "")[:120] + ("..." if conn and len(conn)>120 else ""))

# if not conn:
#     print("ERROR: MONGO_CONNECTION_STRING not set. Check your .env file.")
#     raise SystemExit(1)

# try:
#     client = MongoClient(conn, server_api=ServerApi("1"), serverSelectionTimeoutMS=8000)
#     client.admin.command("ping")
#     print("✅ Successfully connected to MongoDB!")
#     print("Databases:", client.list_database_names())
# except Exception as e:
#     print("❌ Connection failed.")
#     print("Exception type:", type(e).__name__)
#     print("Exception message:", str(e))
#     print("\n--- Full traceback ---")
#     traceback.print_exc()



# quick_count_test.py
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os, traceback
load_dotenv()
conn = os.getenv("MONGO_CONNECTION_STRING")
try:
    client = MongoClient(conn, server_api=ServerApi("1"))
    db = client["ideacritic_db"]
    col = db["debates"]
    print("count_documents:", col.count_documents({}))
except Exception as e:
    print("Error:", type(e).__name__, e)
    traceback.print_exc()
