import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in .env")

client = MongoClient(MONGO_URI)
db = client["idea_critic_db"]
collection = db["rounds"]

def fetch_debates(idea: str):
    """Fetch all debate rounds for a given idea, sorted by round."""
    results = collection.find({"idea": idea}).sort("round", 1)
    for r in results:
        print(f"\n--- Round {r['round']} ---")
        print(f"Optimist: {r['optimist']}")
        print(f"Critic: {r['critic']}")

if __name__ == "__main__":
    idea = input("Enter the idea to fetch debates for:\n> ")
    fetch_debates(idea)
