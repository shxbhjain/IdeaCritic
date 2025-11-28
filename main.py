import os
from typing import List, Dict, Optional
from openai import OpenAI
from pymongo import MongoClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
from urllib.parse import quote_plus

# ================= Load environment =================
load_dotenv()

# ================= MongoDB Setup =================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Set your MONGO_URI in .env")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["idea_critic_db"]
collection = db["rounds"]

# ================= OpenRouter Setup =================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Set your OPENROUTER_API_KEY in .env")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")

# ================= Schemas =================
class OptimistOutput(BaseModel):
    response: str = Field(..., description="Optimist's perspective")

class CriticOutput(BaseModel):
    response: str = Field(..., description="Critic's perspective")

# ================= Prompts =================
OPTIMIST_SYSTEM = """You are the Optimist Agent. 
Highlight strengths, opportunities, and potential of the idea.
Be specific, encouraging, and forward-looking.
"""
CRITIC_SYSTEM = """You are the Critic Agent. 
Point out weaknesses, risks, feasibility issues, and threats in the idea.
Be realistic, constructive, and specific.
"""

# ================= Chat Function =================
def chat(messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# ================= Round Simulation =================
def run_rounds(idea: str, num_rounds: int = 3, context: Optional[str] = None):
    rounds = []
    optimist_history = [{"role": "system", "content": OPTIMIST_SYSTEM}]
    critic_history = [{"role": "system", "content": CRITIC_SYSTEM}]

    # First input (idea intro)
    optimist_history.append({"role": "user", "content": f"Startup Idea: {idea}\n\nContext: {context or ''}"})
    critic_history.append({"role": "user", "content": f"Startup Idea: {idea}\n\nContext: {context or ''}"})

    for r in range(1, num_rounds + 1):
        # Optimist speaks
        opt_reply = chat(optimist_history)
        optimist_output = OptimistOutput(response=opt_reply)

        # Critic responds
        critic_history.append({"role": "assistant", "content": opt_reply})
        crt_reply = chat(critic_history)
        critic_output = CriticOutput(response=crt_reply)

        # Update Optimist history with Critic’s reply for next round
        optimist_history.append({"role": "assistant", "content": crt_reply})

        # Prepare MongoDB entry
        round_entry = {
            "round": r,
            "optimist": opt_reply,
            "critic": crt_reply,
            "idea": idea,
            "context": context,
            "timestamp": datetime.utcnow()
        }

        # Avoid duplicate insertion
        if not collection.find_one({"idea": idea, "round": r}):
            collection.insert_one(round_entry)

        # Save in list (for direct output)
        rounds.append(round_entry)

    return rounds

# ================= Run Example =================
if __name__ == "__main__":
    idea = input("Paste your startup idea:\n> ")
    context = input("\nOptional context (press Enter to skip):\n> ") or None
    num_rounds = int(input("\nHow many rounds? (default 3):\n> ") or 3)

    results = run_rounds(idea, num_rounds, context)

    print("\n=== Debate Results ===")
    for r in results:
        print(f"\n--- Round {r['round']} ---")
        print(f"Timestamp: {r['timestamp']}")
        print(f"Optimist: {r['optimist']}")
        print(f"Critic: {r['critic']}")
