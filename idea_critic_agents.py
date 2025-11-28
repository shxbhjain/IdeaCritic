# import os
# from typing import List, Optional, Dict, Any
# from pydantic import BaseModel, Field
# from openai import OpenAI

# # ========= Provider setup =========
# # Default: OpenAI. To use OpenRouter, set USE_OPENROUTER=1 and OPENROUTER_API_KEY.
# USE_OPENROUTER = os.getenv("USE_OPENROUTER", "0") == "1"

# if USE_OPENROUTER:
#     # OpenRouter
#     if not os.getenv("OPENROUTER_API_KEY"):
#         raise RuntimeError("Set OPENROUTER_API_KEY to use OpenRouter.")
#     client = OpenAI(
#         api_key=os.environ["OPENROUTER_API_KEY"],
#         base_url="https://openrouter.ai/api/v1",
#     )
#     DEFAULT_MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")  # pick any OpenRouter-supported model
# else:
#     # OpenAI
#     if not os.getenv("OPENAI_API_KEY"):
#         raise RuntimeError("Set OPENAI_API_KEY to use OpenAI.")
#     client = OpenAI()
#     DEFAULT_MODEL = os.getenv("MODEL", "gpt-4o-mini")  # fast & capable; change as you like

# # ========= Shared utilities =========

# def chat_json(messages: List[Dict[str, str]], model: Optional=str(DEFAULT_MODEL)) -> Dict[str, Any]:
#     """
#     Calls the chat API and asks for JSON output. If the model supports JSON mode,
#     we request it; otherwise we enforce via instructions.
#     """
#     try:
#         resp = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             # JSON mode for OpenAI & many OpenRouter models:
#             response_format={"type": "json_object"},
#             temperature=0.6,
#         )
#         content = resp.choices[0].message.content
#     except Exception as e:
#         raise RuntimeError(f"LLM call failed: {e}")
#     import json
#     try:
#         return json.loads(content)
#     except Exception:
#         # Fallback: try to extract JSON-ish content
#         import re, json
#         m = re.search(r"\{.*\}", content, re.S)
#         if not m:
#             raise RuntimeError("Model did not return JSON. Content:\n" + content)
#         return json.loads(m.group(0))

# # ========= Schemas (optional but useful) =========

# class OptimistOutput(BaseModel):
#     headline: str = Field(..., description="Punchy one-line potential summary")
#     top_strengths: List[str] = Field(..., description="3–6 strengths")
#     market_opportunities: List[str] = Field(..., description="Specific market/segment opportunities")
#     innovation_score: float = Field(..., ge=0, le=10, description="0–10 novelty/edge")
#     go_to_market: List[str] = Field(..., description="Concrete GTM ideas (channels, early adopters)")
#     key_metrics_to_track: List[str] = Field(..., description="Measurable KPIs for early validation")

# class CriticOutput(BaseModel):
#     headline: str = Field(..., description="Punchy risk summary")
#     top_weaknesses: List[str] = Field(..., description="3–6 weaknesses or gaps")
#     feasibility_risks: List[str] = Field(..., description="Tech/ops feasibility concerns")
#     competitive_threats: List[str] = Field(..., description="Named or archetypal competitors/threats")
#     ethical_legal_flags: List[str] = Field(..., description="Privacy, bias, compliance, safety concerns")
#     failure_modes: List[str] = Field(..., description="How/why it could realistically fail")
#     suggested_mitigations: List[str] = Field(..., description="Concrete, testable mitigations")

# # ========= System prompts =========

# OPTIMIST_SYSTEM = """You are the Optimist Agent for a startup idea evaluation panel.
# Your job is to highlight potential, clear upside, and practical paths to success.
# Be credible: ground claims in plausible mechanisms, not hype. Keep tone upbeat but specific.
# ALWAYS return compact JSON matching the requested schema.
# """

# CRITIC_SYSTEM = """You are the Critic Agent for a startup idea evaluation panel.
# Your job is to identify risks, flaws, blind spots, and realistic failure modes.
# Be constructive: propose concrete mitigations, risk tests, or scope reductions.
# ALWAYS return compact JSON matching the requested schema.
# """

# # ========= Prompt builders =========

# def build_optimist_user_prompt(idea: str, context: Optional[str] = None) -> str:
#     return f"""
# Startup Idea:
# \"\"\"{idea.strip()}\"\"\"

# Context (optional, may include target users, region, constraints):
# \"\"\"{(context or '').strip()}\"\"\"

# Return valid JSON with keys:
# - headline (string)
# - top_strengths (array)
# - market_opportunities (array)
# - innovation_score (number 0-10)
# - go_to_market (array)
# - key_metrics_to_track (array)
# """

# def build_critic_user_prompt(idea: str, context: Optional[str] = None) -> str:
#     return f"""
# Startup Idea:
# \"\"\"{idea.strip()}\"\"\"

# Context (optional, may include target users, region, constraints):
# \"\"\"{(context or '').strip()}\"\"\"

# Return valid JSON with keys:
# - headline (string)
# - top_weaknesses (array)
# - feasibility_risks (array)
# - competitive_threats (array)
# - ethical_legal_flags (array)
# - failure_modes (array)
# - suggested_mitigations (array)
# """

# # ========= Public functions =========

# def run_optimist(idea: str, context: Optional[str] = None, model: Optional[str] = None) -> OptimistOutput:
#     messages = [
#         {"role": "system", "content": OPTIMIST_SYSTEM},
#         {"role": "user", "content": build_optimist_user_prompt(idea, context)},
#     ]
#     raw = chat_json(messages, model or DEFAULT_MODEL)
#     return OptimistOutput(**raw)

# def run_critic(idea: str, context: Optional[str] = None, model: Optional[str] = None) -> CriticOutput:
#     messages = [
#         {"role": "system", "content": CRITIC_SYSTEM},
#         {"role": "user", "content": build_critic_user_prompt(idea, context)},
#     ]
#     raw = chat_json(messages, model or DEFAULT_MODEL)
#     return CriticOutput(**raw)

# # ========= Example CLI usage =========
# if __name__ == "__main__":
#     sample_idea = input("Paste your startup idea:\n> ")
#     sample_ctx = input("\nOptional context (press Enter to skip):\n> ") or None

#     print("\n=== Optimist Agent ===")
#     opt = run_optimist(sample_idea, sample_ctx)
#     print(opt.model_dump_json(indent=2))

#     print("\n=== Critic Agent ===")
#     crt = run_critic(sample_idea, sample_ctx)
#     print(crt.model_dump_json(indent=2))


import os
import sys
import json
import time
import datetime
import traceback
import re
from typing import List, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

# AI + DB libs
import google.generativeai as genai
from pymongo import MongoClient, DESCENDING
from pymongo.server_api import ServerApi

# RAG helper
from rag_store import get_rag_store, index_documents, score_idea

# ---------------------
# Basic setup
# ---------------------
st.set_page_config(page_title="IdeaCritic", page_icon="🚀", layout="wide")
load_dotenv()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
LOCAL_DB_FILE = os.path.join(DATA_DIR, "local_db.json")
PROMPT_CHAR_LIMIT = 16000
DEFAULT_EMB_TOP_K = 6

# ---------------------
# RAG init
# ---------------------
rag_store = get_rag_store()
HAS_RAG = rag_store is not None

# ---------------------
# Gemini model
# ---------------------
@st.cache_resource
def get_gemini_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in .env.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

try:
    gemini_model = get_gemini_model()
except Exception as e:
    gemini_model = None
    st.sidebar.error(f"Model init failed: {e}")

def genai_generate(prompt: str, stream: bool = False, temperature: float = 0.2):
    if gemini_model is None:
        raise RuntimeError("Gemini model not initialized.")
    try:
        if stream:
            return gemini_model.generate_content(prompt, stream=True, temperature=temperature)
        else:
            return gemini_model.generate_content(prompt, temperature=temperature)
    except Exception as e:
        raise RuntimeError(f"Model error: {e}")

# ---------------------
# MongoDB setup with diagnostics and fallback
# ---------------------
@st.cache_resource
def get_mongo_client():
    load_dotenv()
    conn = os.getenv("MONGO_CONNECTION_STRING")
    if not conn:
        raise RuntimeError("MONGO_CONNECTION_STRING missing.")
    client = MongoClient(conn, server_api=ServerApi("1"), serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    return client

try:
    mongo_client = get_mongo_client()
    db = mongo_client.get_database("ideacritic_db")
    debates_collection = db["debates"]
    st.sidebar.success("✅ MongoDB connected!")
except Exception as e:
    mongo_client = None
    debates_collection = None
    st.sidebar.error(f"DB connection failed: {e}")
    st.sidebar.info("Using local JSON fallback storage.")

# ---------------------
# Local JSON fallback
# ---------------------
def load_local_db():
    if not os.path.exists(LOCAL_DB_FILE):
        return []
    try:
        with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_local_doc(doc):
    data = load_local_db()
    oid = str(int(time.time() * 1000))
    doc_copy = dict(doc)
    doc_copy["_id"] = oid
    data.insert(0, doc_copy)
    with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, indent=2)
    return doc_copy

# ---------------------
# Helpers
# ---------------------
def truncate_text(text, limit=PROMPT_CHAR_LIMIT):
    if not text or len(text) <= limit:
        return text
    return text[:limit//2] + "\n...TRUNCATED...\n" + text[-limit//2:]

def stream_response(prompt):
    try:
        for chunk in genai_generate(prompt, stream=True):
            if hasattr(chunk, "text"):
                yield chunk.text
    except Exception as e:
        yield f"[Error: {e}]"

def display_stream(gen):
    placeholder = st.empty()
    result = ""
    for token in gen:
        result += token
        placeholder.markdown(result)
    return result

# ---------------------
# Clarifying questions
# ---------------------
def generate_questions(title, desc):
    prompt = f"""You are a startup mentor. Generate 3–5 clarifying questions for the startup idea below.\nTitle: {title}\nDescription: {desc}"""
    try:
        resp = genai_generate(prompt)
        text = getattr(resp, "text", "")
    except Exception as e:
        st.warning(f"Failed to generate questions: {e}")
        text = ""
    return re.findall(r"Q\d+:\s*(.+)", text) or ["Who is the target customer?", "What is the problem you're solving?", "How do you plan to make money?"]

# ---------------------
# Analysis Page
# ---------------------
def show_new_analysis_page():
    st.title("🚀 New Idea Analysis")
    title = st.text_input("Enter a short title", placeholder="e.g., AI-powered fitness coach")
    desc = st.text_area("Describe your startup idea", height=150)

    if st.button("Proceed", type="primary"):
        if not title or not desc:
            st.error("Please fill in both fields.")
        else:
            st.session_state["title"] = title
            st.session_state["desc"] = desc
            st.session_state["questions"] = generate_questions(title, desc)
            st.experimental_rerun()

    if "questions" in st.session_state:
        st.subheader("📝 Clarifying Questions")
        answers = {}
        for i, q in enumerate(st.session_state["questions"], 1):
            answers[f"Q{i}"] = st.text_input(q, key=f"ans{i}")

        num_rounds = st.slider("Number of debate rounds", 1, 5, 3)

        if HAS_RAG:
            score = score_idea(title, desc)
            st.metric("💡 Market Fit Score", f"{score['score']} / 100")
            st.caption(score["explanation"])

        if st.button("Start Analysis", type="primary"):
            idea_context = desc + "\n\n" + "\n".join(f"{k}: {v}" for k, v in answers.items())
            transcript = ""
            last_resp = ""

            for i in range(num_rounds):
                st.markdown(f"### Round {i+1}")

                with st.spinner("Optimist is thinking..."):
                    opt_prompt = f"You are a startup optimist. Respond in 3 bullet points to the idea: {idea_context}. Previous statement: {last_resp}"
                    opt_text = display_stream(stream_response(opt_prompt))
                    transcript += f"\nRound {i+1} - Optimist: {opt_text}"
                    last_resp = opt_text

                with st.spinner("Critic is thinking..."):
                    crt_prompt = f"You are a startup critic. Respond in 3 bullet points to the idea: {idea_context}. Previous statement: {last_resp}"
                    crt_text = display_stream(stream_response(crt_prompt))
                    transcript += f"\nRound {i+1} - Critic: {crt_text}"
                    last_resp = crt_text

            with st.spinner("Summarizing discussion..."):
                summary_prompt = f"Summarize this startup discussion:\n{transcript}"
                summary = display_stream(stream_response(summary_prompt))

            doc = {
                "idea_title": title,
                "idea_description": desc,
                "clarifying_answers": answers,
                "debate_transcript": transcript,
                "final_summary": summary,
                "created_at": datetime.datetime.now(datetime.timezone.utc),
            }

            if debates_collection:
                res = debates_collection.insert_one(doc)
                st.success(f"Saved to MongoDB (ID: {res.inserted_id})")
            else:
                save_local_doc(doc)
                st.info("Saved locally (no DB connection)")

            if HAS_RAG:
                index_documents([{"id": title, "text": desc + transcript, "source": title}])
                st.info("Indexed to RAG store.")

# ---------------------
# History Page
# ---------------------
def show_history_page():
    st.title("📚 History")
    data = []
    if debates_collection:
        data = list(debates_collection.find().sort("created_at", DESCENDING))
    else:
        data = load_local_db()

    if not data:
        st.warning("No analyses found.")
        return

    for d in data:
        with st.expander(d.get("idea_title", "Untitled")):
            st.markdown(d.get("final_summary", "No summary."))

# ---------------------
# Sidebar Navigation
# ---------------------
page = st.sidebar.radio("Navigation", ["New Analysis", "History"])

if page == "New Analysis":
    show_new_analysis_page()
else:
    show_history_page()
