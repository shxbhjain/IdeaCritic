# bootstrap_rag.py
from pymongo import MongoClient
from rag_store import get_rag_store, index_from_db
import os

MONGO_CONN = os.environ.get("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_CONN)
db = client['ideacritic_db']
collection = db['debates']

index_from_db(collection)
print("Indexing finished")
