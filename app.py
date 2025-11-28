import os
import re
import json
import time
import datetime
import traceback
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv


from langchain_core.prompts import PromptTemplate
try:
    from langchain.chains import LLMChain
except Exception:
    class LLMChain:
        def __init__(self, llm, prompt):
            self.llm = llm
            self.prompt = prompt
        def invoke(self, inputs):
            prompt_text = self.prompt.format(**inputs)
            if hasattr(self.llm, "invoke"):
                out = self.llm.invoke(prompt_text)
                content = getattr(out, "content", None)
                if content is None and hasattr(out, "generations"):
                    try:
                        content = out.generations[0][0].text
                    except Exception:
                        pass
                return {"text": content if content is not None else str(out)}
            if hasattr(self.llm, "predict"):
                return {"text": self.llm.predict(prompt_text)}
            if callable(self.llm):
                return {"text": self.llm(prompt_text)}
            return {"text": prompt_text}
        def run(self, inputs):
            res = self.invoke(inputs)
            return res.get("text", "")


try:
    from rag_store import get_rag_store, index_documents, score_idea
except Exception:
    def get_rag_store():
        return None
    def index_documents(docs: List[Dict]):
        return None
    def score_idea(title: str, desc: str) -> Dict:
        return {"score": 0.0, "explanation": "RAG disabled (rag_store not available)"}


load_dotenv()
st.set_page_config(page_title="IdeaCritic (RAG)", page_icon="🚀", layout="wide")

def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

LLM_BACKEND = None
LLM_OBJ = None


if GOOGLE_API_KEY and LLM_OBJ is None:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        LLM_OBJ = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
            api_key=GOOGLE_API_KEY,
            streaming=False
        )
        LLM_BACKEND = "google_gemini"
    except Exception:
        LLM_OBJ = None
        LLM_BACKEND = None

# 2) Local LlamaCpp (0.2+ path first, then legacy)
if LLM_OBJ is None and LOCAL_LLM_PATH:
    LlamaCpp = None
    try:
        from langchain_community.llms import LlamaCpp as _LlamaCpp  # 0.2+
        LlamaCpp = _LlamaCpp
    except Exception:
        try:
            from langchain.llms import LlamaCpp as _LlamaCpp  # legacy
            LlamaCpp = _LlamaCpp
        except Exception:
            LlamaCpp = None
    if LlamaCpp is not None:
        try:
            LLM_OBJ = LlamaCpp(model_path=LOCAL_LLM_PATH, n_ctx=2048)
            LLM_BACKEND = "llama_cpp"
        except Exception:
            LLM_OBJ = None
            LLM_BACKEND = None

# 3) Dummy fallback LLM
if LLM_OBJ is None:
    class DummyLLM:
        def predict(self, prompt: str) -> str:
            p = prompt.lower()
            if "clarifying questions" in p or "produce exactly 3–5" in p or "produce exactly 3-5" in p:
                return (
                    "1. Who is the primary target user and what specific scenario do they face?\n"
                    "2. What problem does this solve better than current workarounds or competitors?\n"
                    "3. How will you acquire the first 100 paying users and through which channel?\n"
                    "4. What is the core differentiator that’s hard to copy in 6–12 months?\n"
                    "5. How will you price this initially and what are the expected unit economics?"
                )
            if "you are a startup optimist" in p:
                return "- Large addressable market.\n- Early signals of demand.\n- Low-cost MVP path."
            if "you are a startup critic" in p:
                return "- Strong incumbents.\n- CAC vs LTV unclear.\n- Scaling needs partnerships."
            if "expert business analyst" in p or "final actionable summary" in p:
                return "Verdict: Promising but risky.\n\n- Validate with pilots.\n- Prove unit economics.\n- Secure distribution."
            return "Fallback response (dummy)."
        def __call__(self, prompt: str, **kwargs):
            return self.predict(prompt)
        def predict_batch(self, prompts: List[str]) -> List[str]:
            return [self.predict(p) for p in prompts]
        def generate(self, prompts):
            return [self.predict(p) for p in prompts]
    LLM_OBJ = DummyLLM()
    LLM_BACKEND = "dummy"

# Sidebar: backend status
st.sidebar.markdown("## 🔧 LLM Backend")
if LLM_BACKEND == "google_gemini":
    st.sidebar.success("Google Gemini active (langchain-google-genai)")
elif LLM_BACKEND == "llama_cpp":
    st.sidebar.success("Local LlamaCpp model loaded")
else:
    st.sidebar.warning("Using Dummy LLM fallback")
    if not LOCAL_LLM_PATH and not GOOGLE_API_KEY:
        st.sidebar.info("Set LOCAL_LLM_PATH or GOOGLE_API_KEY in .env to use a real model.")


# RAG Initialization

rag_store = get_rag_store()
HAS_RAG = rag_store is not None
st.sidebar.markdown("## 📚 RAG Store")
if HAS_RAG:
    try:
        total = getattr(rag_store, "index", None)
        n = total.ntotal if total is not None else None
        if n is None and hasattr(rag_store, "ntotal"):
            n = rag_store.ntotal
        if n is None:
            st.sidebar.success("RAG store loaded.")
        else:
            st.sidebar.success(f"RAG store loaded. {n} items indexed.")
    except Exception:
        st.sidebar.success("RAG store loaded.")
else:
    st.sidebar.warning("RAG dependencies not found (faiss/sentence-transformers).")


clarify_prompt = PromptTemplate(
    input_variables=["title", "desc"],
    template="""
You are a startup mentor. A founder provided this idea:
Title: {title}
Description: {desc}

Produce exactly 3–5 clarifying questions to better understand this idea.

FORMAT RULES (follow strictly):
- Return a plain numbered list, one per line. Use only this structure:
  1. <question>
  2. <question>
  3. <question>
- Do not include generic placeholders like "Question one/two/three".
- Do not use XML/HTML/JSON or any tags.
- Each item must be a single, specific sentence ending with a question mark.
- Focus on market, target user, differentiation, feasibility, and go-to-market.
"""
)

optimist_prompt = PromptTemplate(
    input_variables=["idea", "transcript"],
    template="""
You are a startup Optimist. The startup idea is: "{idea}".

Debate so far:
{transcript}

Now respond with exactly 3 concise bullet points.
- Defend against the Critic’s previous objections where possible.
- Highlight new strengths and opportunities.
- Keep the points short, sharp, and positive.
"""
)

critic_prompt = PromptTemplate(
    input_variables=["idea", "transcript"],
    template="""
You are a startup Critic. The startup idea is: "{idea}".

Debate so far:
{transcript}

The Optimist has just spoken. Now respond point-by-point:
- Mirror each Optimist bullet point with a Critic counterpoint.
- Keep the order aligned (1 vs 1, 2 vs 2, etc).
- Use short and sharp sentences that directly challenge optimism.
"""
)

summary_prompt = PromptTemplate(
    input_variables=["idea", "transcript"],
    template="""
You are an expert Business Analyst. You have a discussion transcript for the startup idea "{idea}".

Discussion Transcript:
---
{transcript}
---

Write a final actionable summary:
- First, a short paragraph with your verdict.
- Then 3 key bullet points.
"""
)


# Chain utilities

def run_chain_safe(chain: LLMChain, inputs: Dict) -> str:
    try:
        if hasattr(chain, "invoke"):
            result = chain.invoke(inputs)
            if isinstance(result, dict):
                txt = result.get("text")
                if txt is not None:
                    return txt
            if hasattr(result, "content"):
                return str(result.content)
            return str(result)
        return chain.run(inputs)
    except Exception:
        prompt_text = chain.prompt.format(**inputs)
        llm = getattr(chain, "llm", None)
        if hasattr(llm, "invoke"):
            try:
                out = llm.invoke(prompt_text)
                return getattr(out, "content", str(out))
            except Exception:
                pass
        if hasattr(llm, "predict"):
            try:
                return llm.predict(prompt_text)
            except Exception:
                pass
        if callable(llm):
            try:
                return llm(prompt_text)
            except Exception:
                pass
        if hasattr(llm, "generate"):
            try:
                res = llm.generate([prompt_text])
                if isinstance(res, list) and res:
                    return res[0]
                if hasattr(res, "generations"):
                    return res.generations[0][0].text
                return str(res)
            except Exception:
                pass
        return prompt_text

# Build chains

clarify_chain = LLMChain(llm=LLM_OBJ, prompt=clarify_prompt)
optimist_chain = LLMChain(llm=LLM_OBJ, prompt=optimist_prompt)
critic_chain = LLMChain(llm=LLM_OBJ, prompt=critic_prompt)
summary_chain = LLMChain(llm=LLM_OBJ, prompt=summary_prompt)


# MongoDB connection + diagnostics

from pymongo import MongoClient, DESCENDING
from pymongo.server_api import ServerApi

def get_mongo_client_from_env() -> Optional[MongoClient]:
    conn = os.getenv("MONGO_CONNECTION_STRING", "").strip()
    if not conn:
        return None
    try:
        client = MongoClient(conn, server_api=ServerApi("1"), serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        return client
    except Exception:
        return None

mongo_client = get_mongo_client_from_env()
st.sidebar.markdown("## 💾 Storage")
if mongo_client:
    db = mongo_client.get_database("ideacritic_db")
    debates_collection = db.get_collection("debates")
    st.sidebar.success("✅ MongoDB connected")
else:
    debates_collection = None
    st.sidebar.info("MongoDB not configured or not reachable — using local JSON fallback")


# Persistence utilities

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
LOCAL_DB_FILE = os.path.join(DATA_DIR, "local_db.json")

def save_doc(doc: Dict) -> Dict:
    doc_copy = dict(doc)
    if not isinstance(doc_copy.get("created_at"), datetime.datetime):
        doc_copy["created_at"] = datetime.datetime.now(datetime.timezone.utc)

    if debates_collection is not None:
        res = debates_collection.insert_one(doc_copy)
        return {"storage": "mongo", "inserted_id": str(res.inserted_id)}

    data = []
    if os.path.exists(LOCAL_DB_FILE):
        try:
            with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    doc_copy["_id"] = str(int(time.time() * 1000))
    data.insert(0, doc_copy)
    with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, indent=2)
    return {"storage": "local", "inserted_id": doc_copy["_id"]}

def load_all_analyses() -> List[Dict]:
    if debates_collection is not None:
        try:
            return list(debates_collection.find().sort("created_at", DESCENDING))
        except Exception:
            return []
    if os.path.exists(LOCAL_DB_FILE):
        try:
            with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# Clarifying-questions generator (robust + fallback)

def _fallback_questions(title: str, desc: str) -> List[str]:
    base = (title or "your product").strip()
    return [
        f"Who exactly is the primary target user for {base}, and what narrow use case will you serve first?",
        "What painful, frequent problem does this solve better than the user's current workaround?",
        "Why will users pick this over existing alternatives, and what is the core differentiator?",
        "What is the initial go-to-market channel and how will you acquire your first 100 paying users?",
        "How will you price it and what are the expected unit economics at launch?"
    ]

def _looks_like_placeholder(q: str) -> bool:
    ql = q.strip().lower()
    if re.fullmatch(r"question\s*(one|two|three|four|five)\??", ql):
        return True
    if re.fullmatch(r"<\s*question\s*>.*", ql) or re.fullmatch(r".*</\s*question\s*>", ql):
        return True
    if ql in {"question?", "question"}:
        return True
    return False

def generate_clarifying_questions(title: str, desc: str) -> List[str]:
    raw = run_chain_safe(clarify_chain, {"title": title, "desc": desc}) or ""
    text = raw.strip()

    # 1) Try XML-ish tags
    xml_matches = re.findall(r"<\s*question\s*>(.*?)<\s*/\s*question\s*>", text, flags=re.IGNORECASE | re.DOTALL)
    if xml_matches:
        qs = [re.sub(r"\s+", " ", q).strip() for q in xml_matches]
    else:
        # 2) Numbered or bulleted
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        numbered = [re.sub(r"^\s*\d+[\.\)\-]\s*", "", l) for l in lines if re.match(r"^\s*\d+[\.\)\-]\s*", l)]
        bullets  = [re.sub(r"^\s*[-*•]\s*", "", l) for l in lines if re.match(r"^\s*[-*•]\s*", l)]
        if numbered:
            qs = numbered
        elif bullets:
            qs = bullets
        else:
            # 3) Fallback: split on '?'
            chunks = [c.strip() for c in re.split(r"\?+", text) if c.strip()]
            qs = [c + "?" for c in chunks]

    # Normalize
    cleaned = []
    for q in qs:
        q = re.sub(r"^['\"“”‘’]+|['\"“”‘’]+$", "", q)
        q = re.sub(r"^\s*\d+[\.\)\-]\s*", "", q)
        q = re.sub(r"^\s*[-*•]\s*", "", q)
        q = re.sub(r"</?[^>]+>", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        if q and not q.endswith("?"):
            q += "?"
        if q:
            cleaned.append(q)

    # Filter placeholders / junk, dedupe
    cleaned = [q for q in cleaned if not _looks_like_placeholder(q)]
    cleaned = [q for q in cleaned if len(q) > 10]
    seen, deduped = set(), []
    for q in cleaned:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(q)

    # Ensure 3–5; otherwise fallback
    if len(deduped) >= 5:
        return deduped[:5]
    if 3 <= len(deduped) <= 5:
        return deduped
    return _fallback_questions(title, desc)[:5]

# ---------------------------
# Helper wrappers
# ---------------------------
def get_agent_response(chain: LLMChain, idea: str, transcript: str) -> str:
    return run_chain_safe(chain, {"idea": idea, "transcript": transcript})

def get_summary(idea: str, transcript: str) -> str:
    return run_chain_safe(summary_chain, {"idea": idea, "transcript": transcript})

# ---------------------------
# Pages
# ---------------------------
def show_new_analysis_page():
    st.title("🚀 New Idea Analysis")

    if "clarifying_questions" not in st.session_state:
        st.header("Step 1: Describe your startup idea")
        startup_idea_title = st.text_input(
            "Enter a short title for your idea",
            placeholder="e.g., AI-powered fitness coach"
        )
        startup_idea_desc = st.text_area(
            "Describe your startup idea in detail",
            placeholder="My startup will offer personalized workout and meal plans...",
            height=150
        )

        if st.button("Proceed", type="primary"):
            if not startup_idea_title or not startup_idea_desc:
                st.error("Please enter both a title and a description.")
            else:
                with st.spinner("Generating clarifying questions..."):
                    st.session_state["clarifying_questions"] = generate_clarifying_questions(
                        startup_idea_title, startup_idea_desc
                    )
                st.session_state["idea_title"] = startup_idea_title
                st.session_state["idea_desc"] = startup_idea_desc
                st.session_state["answers"] = {}
                _safe_rerun()
    else:
        st.header(f"Analyzing: {st.session_state.get('idea_title', 'Your Idea')}")
        st.subheader("Step 2: Answer the clarifying questions")
        for i, q in enumerate(st.session_state.get("clarifying_questions", []), start=1):
            st.session_state["answers"][f"Q{i}"] = st.text_input(q, key=f"q{i}")

        st.divider()
        st.subheader("Step 3: Start the analysis")
        num_rounds = st.slider("How many rounds should the discussion be?", 1, 5, 3)

        # --- RAG Integration: Scoring ---
        if HAS_RAG:
            st.markdown("#### 💡 RAG Market Fit Score")
            with st.spinner("Scoring idea against RAG store..."):
                try:
                    score_data = score_idea(
                        st.session_state.get("idea_title", ""),
                        st.session_state.get("idea_desc", "")
                    )
                except Exception as e:
                    score_data = {"score": 0.0, "error": str(e)}
                st.metric("Score (0-100)", f"{score_data.get('score', 0.0)}")
                with st.expander("View RAG score explanation"):
                    st.json(score_data)
        # --- End RAG ---

        if st.button("Start Analysis", type="primary"):
            st.session_state["running_analysis"] = True
            idea_full_context = st.session_state["idea_desc"] + "\n\nAnswers:\n"
            for i, q in enumerate(st.session_state.get("clarifying_questions", []), start=1):
                ans = st.session_state["answers"].get(f"Q{i}", "Not answered")
                idea_full_context += f"Q: {q}\nA: {ans}\n"

            conversation_history_for_db = ""
            last_response = ""

            st.subheader("💬 Live Discussion Transcript")
            for i in range(num_rounds):
                round_number = i + 1
                st.markdown(f"#### Round {round_number}")

                st.markdown("*Optimist's Turn:*")
                with st.spinner("Optimist is thinking..."):
                    optimist_response = get_agent_response(optimist_chain, idea_full_context, last_response)
                    st.markdown(optimist_response)
                conversation_history_for_db += f"\nRound {round_number} - Optimist: {optimist_response}"
                last_response = optimist_response

                st.divider()

                st.markdown("*Critic's Turn:*")
                with st.spinner("Critic is thinking..."):
                    critic_response = get_agent_response(critic_chain, idea_full_context, last_response)
                    st.markdown(critic_response)
                conversation_history_for_db += f"\nRound {round_number} - Critic: {critic_response}"
                last_response = critic_response

            st.divider()
            st.subheader("--- Final Business Analyst Summary ---")
            with st.spinner("Drafting the final summary..."):
                final_summary = get_summary(idea_full_context, conversation_history_for_db)
                st.markdown(final_summary)

            doc = {
                "idea_title": st.session_state.get("idea_title"),
                "idea_description": st.session_state.get("idea_desc"),
                "clarifying_answers": st.session_state.get("answers"),
                "debate_transcript": conversation_history_for_db.strip(),
                "final_summary": final_summary,
                "created_at": datetime.datetime.now(datetime.timezone.utc)
            }

            try:
                saved = save_doc(doc)
                st.success(f"💾 Analysis saved to {saved['storage']} (id: {saved['inserted_id']})")

                if HAS_RAG:
                    with st.spinner("Indexing document to RAG store..."):
                        try:
                            index_documents([{
                                "id": saved.get('inserted_id', st.session_state.get("idea_title")),
                                "text": idea_full_context + "\n\n" + conversation_history_for_db + "\n\n" + final_summary,
                                "source": st.session_state.get("idea_title")
                            }])
                            st.info("✅ Idea indexed to RAG store.")
                        except Exception as e:
                            st.warning(f"Failed to index in RAG store: {e}")

            except Exception as e:
                st.error(f"❌ Failed to save analysis: {e}")
            finally:
                st.session_state["running_analysis"] = False

def show_analysis_history_page(all_analyses: List[Dict]):
    st.title("📚 Analysis Archive")
    if "selected_debate_id" in st.session_state:
        sel = st.session_state.selected_debate_id
        selected_analysis = next(
            (d for d in all_analyses if str(d.get("_id")) == sel or str(d.get("_id")) == str(sel)),
            None
        )
        if selected_analysis:
            if st.button("⬅️ Back to Archive"):
                del st.session_state["selected_debate_id"]
                _safe_rerun()
            st.header(f"Viewing Analysis: {selected_analysis.get('idea_title')}")
            created = selected_analysis.get("created_at")
            try:
                if isinstance(created, str):
                    try:
                        created = datetime.datetime.fromisoformat(created)
                    except ValueError:
                        pass
                cap = created.strftime("%B %d, %Y at %I:%M %p") if isinstance(created, datetime.datetime) else str(created)
            except Exception:
                cap = str(created)
            st.caption(f"Analyzed on: {cap}")
            st.divider()
            st.subheader("Final Business Analyst Summary")
            st.markdown(selected_analysis.get("final_summary", "No summary available."))
            st.subheader("Full Analysis Breakdown")
            with st.expander("Original Idea Description"):
                st.write(selected_analysis.get("idea_description", ""))
            with st.expander("Clarifying Answers"):
                st.write(selected_analysis.get("clarifying_answers", {}))
            with st.expander("Full Discussion Transcript (Formatted)"):
                st.text(selected_analysis.get("debate_transcript", ""))
        else:
            st.error("Could not find the selected analysis.")
            if st.button("⬅️ Back to Archive"):
                del st.session_state["selected_debate_id"]
                _safe_rerun()
    else:
        if not all_analyses:
            st.warning("Your archive is empty. Run an analysis first.")
            return
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Total Analyses", len(all_analyses))
        try:
            most_recent = all_analyses[0].get("created_at")
            if isinstance(most_recent, str):
                try:
                    most_recent = datetime.datetime.fromisoformat(most_recent)
                except ValueError:
                    pass
            most_recent_str = most_recent.strftime("%B %d, %Y") if isinstance(most_recent, datetime.datetime) else str(most_recent)
        except Exception:
            most_recent_str = "Unknown"
        col2.metric("Most Recent", most_recent_str)
        st.divider()
        st.subheader("All Saved Analyses")
        for analysis in all_analyses:
            try:
                container = st.container(border=True)
            except TypeError:
                container = st.container()
            with container:
                a_col1, a_col2 = st.columns([4, 1])
                with a_col1:
                    st.markdown(f"##### {analysis.get('idea_title','Untitled')}")
                    created = analysis.get("created_at")
                    try:
                        if isinstance(created, str):
                            try:
                                created = datetime.datetime.fromisoformat(created)
                            except ValueError:
                                pass
                        cap = created.strftime("%B %d, %Y at %I:%M %p") if isinstance(created, datetime.datetime) else str(created)
                    except Exception:
                        cap = str(created)
                    st.caption(f"Created on: {cap}")
                with a_col2:
                    if st.button("View Full Report", key=f"view_{analysis.get('_id')}"):
                        st.session_state.selected_debate_id = str(analysis.get('_id'))
                        _safe_rerun()
                summary_preview = analysis.get("final_summary", "No summary available.")
                st.write(summary_preview[:200] + "...")

# ---------------------------
# Sidebar & routing
# ---------------------------
st.sidebar.markdown("## 🚀 IdeaCritic")
page_options = ["New Analysis", "Analysis History"]

def on_page_change():
    if st.session_state.get("radio_nav") == "New Analysis":
        for key in ["clarifying_questions", "idea_title", "idea_desc", "answers", "selected_debate_id", "running_analysis"]:
            if key in st.session_state:
                del st.session_state[key]

selected_page = st.sidebar.radio(
    "Main Menu",
    page_options,
    key="radio_nav",
    on_change=on_page_change,
    label_visibility="collapsed"
)
st.sidebar.divider()

# Show storage stats
try:
    if mongo_client:
        total_analyses = debates_collection.count_documents({})
    else:
        if os.path.exists(LOCAL_DB_FILE):
            with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                total_analyses = len(json.load(f))
        else:
            total_analyses = 0
    st.sidebar.metric("Total Analyses Saved", total_analyses)
except Exception as e:
    st.sidebar.error("DB connection error (detailed):")
    st.sidebar.code(f"{type(e).__name__}: {e}")
    st.sidebar.markdown("**Traceback**")
    st.sidebar.code(traceback.format_exc())

# Load archive if needed
if selected_page == "Analysis History":
    all_analyses = load_all_analyses()
else:
    all_analyses = []

# Route
if selected_page == "New Analysis":
    show_new_analysis_page()
elif selected_page == "Analysis History":
    show_analysis_history_page(all_analyses)
