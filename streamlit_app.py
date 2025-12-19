

import streamlit as st
import os
import datetime
import re
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.server_api import ServerApi

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# RAG imports
try:
    from rag_store import get_rag_store, get_marketplace_context
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IdeaCritic (LangChain)",
    page_icon="🚀",
    layout="wide"
)

load_dotenv()

# --------------------------------------------------
# LLM & DB Setup
# --------------------------------------------------
@st.cache_resource
def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found in .env file. Please add it.")
        st.stop()
    try:
        # return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=google_api_key, streaming=True)
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, streaming=True)
    except Exception as e:
        st.error(f"❌ Failed to configure Google AI: {e}")
        st.stop()


@st.cache_resource
def get_mongo_connection():
    uri = os.getenv("MONGO_CONNECTION_STRING")
    if not uri:
        st.error("MONGO_CONNECTION_STRING missing in .env")
        st.stop()

    client = MongoClient(uri, server_api=ServerApi("1"))
    client.admin.command("ping")
    return client


llm = get_llm()
mongo_client = get_mongo_connection()

db = mongo_client["ideacritic_db"]
debates_collection = db["debates"]

# --------------------------------------------------
# RAG Setup
# --------------------------------------------------
@st.cache_resource
def get_rag_context():
    """Get RAG store for retrieving relevant past analyses."""
    if not HAS_RAG:
        return None
    try:
        return get_rag_store()
    except Exception:
        return None

def initialize_marketplace_data():
    """Initialize the RAG store with sample marketplace data."""
    if not HAS_RAG:
        return
    
    try:
        from rag_store import add_marketplace_data
        
        # Sample marketplace data - in real implementation, this would come from APIs
        sample_marketplace_data = [
            {
                "text": "Fitness tracking apps dominate the market with over 500M downloads. Top apps include MyFitnessPal, Strava, and Nike Training Club. Average revenue per user: $2.50/month. Key features: workout tracking, social challenges, personalized plans.",
                "source": "App Store Analytics",
                "category": "Health  & Fitness",
                "metadata": {"downloads": "500M+", "avg_revenue": "$2.50/user/month"}
            },
            {
                "text": "AI-powered educational apps growing 300% YoY. Duolingo leads with 500M users. Market size: $2.2B. Key success factors: gamification, personalized learning paths, offline access.",
                "source": "Google Play Store",
                "category": "Education",
                "metadata": {"growth": "300%", "market_size": "$2.2B"}
            },
            {
                "text": "Mental health apps market projected to reach $6.2B by 2027. Headspace and Calm dominate with subscription models. Average price: $4.99/month. Focus on mindfulness, CBT, sleep tracking.",
                "source": "Market Research Report",
                "category": "Health & Wellness",
                "metadata": {"projection": "$6.2B by 2027", "avg_price": "$4.99/month"}
            },
            {
                "text": "SaaS productivity tools market: $50B+. Slack, Notion, Trello lead. Freemium model successful. Key features: collaboration, automation, integrations. Customer acquisition cost: $150-300.",
                "source": "Industry Report",
                "category": "Productivity",
                "metadata": {"market_size": "$50B+", "cac": "$150-300"}
            }
        ]
        
        success = add_marketplace_data(sample_marketplace_data)
        if success:
            st.success("📊 Marketplace data loaded into RAG system!")
        else:
            st.warning("⚠️ Could not load marketplace data")
            
    except Exception as e:
        st.warning(f"⚠️ Marketplace data loading failed: {e}")

# Initialize marketplace data on app start
if HAS_RAG and 'marketplace_initialized' not in st.session_state:
    initialize_marketplace_data()
    st.session_state.marketplace_initialized = True

def retrieve_relevant_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant context from past analyses and marketplace data."""
    context_parts = []
    
    # Get past analysis context
    try:
        store = get_rag_store()
        if store:
            analysis_results = store.search(query, top_k=top_k)
            if analysis_results:
                analysis_context = "\n\n--- Relevant Past Analyses ---\n"
                for i, result in enumerate(analysis_results, 1):
                    analysis_context += f"{i}. {result.get('text', 'No text available')}\n"
                    if 'idea_title' in result:
                        analysis_context += f"   Idea: {result['idea_title']}\n"
                    analysis_context += "\n"
                context_parts.append(analysis_context)
    except Exception:
        pass
    
    # Get marketplace context
    marketplace_context = get_marketplace_context(query, top_k=2)
    if marketplace_context:
        context_parts.append(marketplace_context)
    
    return "".join(context_parts)

# --------------------------------------------------
# Prompt Templates
# --------------------------------------------------
clarify_prompt = PromptTemplate(
    input_variables=["title", "desc"],
    template="""
You are a startup mentor.

Title: {title}
Description: {desc}

Generate exactly 3–5 clarifying questions.

Rules:
- Output strictly as numbered list
- Short, precise, no fluff
"""
)

optimist_prompt = PromptTemplate(
    input_variables=["idea", "transcript", "context"],
    template="""
You are a startup Optimist. Use the relevant past analyses to inform your response.

Idea:
{idea}

Relevant Past Analyses:
{context}

Transcript so far:
{transcript}

Respond with exactly 3 bullet points defending the idea, learning from past successful analyses.
"""
)

critic_prompt = PromptTemplate(
    input_variables=["idea", "transcript", "context"],
    template="""
You are a startup Critic. Use the relevant past analyses to inform your response.

Idea:
{idea}

Relevant Past Analyses:
{context}

Transcript so far:
{transcript}

Counter the Optimist point-by-point (same order), drawing from past critical insights.
"""
)

summary_prompt = PromptTemplate(
    input_variables=["idea", "transcript", "context"],
    template="""
You are a Business Analyst. Use relevant past analyses for context.

Idea:
{idea}

Relevant Past Analyses:
{context}

Transcript:
{transcript}

Give:
- One verdict paragraph (considering past patterns)
- 3 actionable bullet points
"""
)

# --------------------------------------------------
# Chains (LCEL)
# --------------------------------------------------
clarify_chain = clarify_prompt | llm
optimist_chain = optimist_prompt | llm
critic_chain = critic_prompt | llm
summary_chain = summary_prompt | llm

# --------------------------------------------------
# Helpers (CRITICAL FIX)
# --------------------------------------------------
def extract_text(response):
    return response.content if hasattr(response, "content") else str(response)


@st.cache_data
def generate_clarifying_questions(title, desc):
    resp = clarify_chain.invoke({"title": title, "desc": desc})
    text = extract_text(resp)

    questions = [
        q.strip() for q in text.split("\n")
        if re.match(r"^\d+\.", q.strip())
    ]
    return questions if questions else [text]


def agent_response(chain, idea, transcript, context=""):
    resp = chain.invoke({"idea": idea, "transcript": transcript, "context": context})
    return extract_text(resp)


def get_summary(idea, transcript, context=""):
    resp = summary_chain.invoke({"idea": idea, "transcript": transcript, "context": context})
    return extract_text(resp)

# --------------------------------------------------
#  Page: New Analysis
# --------------------------------------------------
def show_new_analysis_page():
    # Modern header
    st.title("🚀 IdeaCritic - AI Startup Analysis")
    st.markdown("***Transform your startup idea into a data-driven opportunity with AI-powered market intelligence***")

    if "clarifying_questions" not in st.session_state:
        title = st.text_input("Startup Title")
        desc = st.text_area("Describe your idea", height=150)

        if st.button("Proceed", type="primary"):
            if not title or not desc:
                st.error("Title and description required.")
                return

            with st.spinner("Generating questions..."):
                st.session_state.clarifying_questions = generate_clarifying_questions(title, desc)
                st.session_state.idea_title = title
                st.session_state.idea_desc = desc
                st.session_state.answers = {}
            st.rerun()

    else:
        st.header("Answer Clarifying Questions")

        for i, q in enumerate(st.session_state.clarifying_questions, 1):
            clean = re.sub(r"^\d+\.\s*", "", q)
            st.session_state.answers[f"Q{i}"] = st.text_area(clean, key=f"q{i}")

        rounds = st.slider("Discussion rounds", 1, 5, 3)

        if st.button("Start Analysis", type="primary"):
            idea_context = st.session_state.idea_desc + "\n\n"
            for i, q in enumerate(st.session_state.clarifying_questions, 1):
                idea_context += f"{q}\nA: {st.session_state.answers.get(f'Q{i}', '')}\n"

            # Retrieve relevant RAG context
            rag_context = retrieve_relevant_context(
                f"{st.session_state.idea_title} {st.session_state.idea_desc}", 
                top_k=3
            )
            
            if rag_context:
                # Check if marketplace data is included
                has_marketplace = "[MARKETPLACE]" in rag_context or "--- Marketplace Intelligence ---" in rag_context
                has_analysis = "--- Relevant Past Analyses ---" in rag_context
                
                status_parts = []
                if has_analysis:
                    status_parts.append("past analyses")
                if has_marketplace:
                    status_parts.append("marketplace data")
                
                if status_parts:
                    st.info(f"🧠 RAG activated: Using insights from {', '.join(status_parts)}")
                else:
                    st.info("🧠 RAG activated: Using historical data")
            else:
                st.warning("📚 No relevant data found in RAG store")

            transcript = ""
            st.subheader("💬 Debate")

            for r in range(rounds):
                st.markdown(f"### Round {r + 1}")

                opt = agent_response(optimist_chain, idea_context, transcript, rag_context)
                st.markdown("**Optimist:**")
                st.markdown(opt)
                transcript += f"\nOptimist: {opt}"

                crit = agent_response(critic_chain, idea_context, transcript, rag_context)
                st.markdown("**Critic:**")
                st.markdown(crit)
                transcript += f"\nCritic: {crit}"

            st.divider()
            final_summary = get_summary(idea_context, transcript, rag_context)
            st.subheader("📌 Final Verdict")
            st.markdown(final_summary)

            debates_collection.insert_one({
                "idea_title": st.session_state.idea_title,
                "idea_description": st.session_state.idea_desc,
                "clarifying_answers": st.session_state.answers,
                "debate_transcript": transcript,
                "final_summary": final_summary,
                "created_at": datetime.datetime.utcnow()
            })

            st.success("Analysis saved successfully.")

# --------------------------------------------------
#  Page: History
# --------------------------------------------------
def show_history_page():
    st.title("📚 Analysis Archive")

    items = list(debates_collection.find().sort("created_at", DESCENDING))
    if not items:
        st.info("No saved analyses.")
        return

    for doc in items:
        with st.expander(doc["idea_title"]):
            st.markdown(doc["final_summary"])

# --------------------------------------------------
# Sidebar (DEFINED ONCE)
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🚀 IdeaCritic")
    st.caption("AI-Powered Startup Analysis")

    # System Status Accordion
    with st.expander("🔧 System Status", expanded=False):
        st.markdown("### 🧠 LLM Backend")
        st.success("Google Gemini active (langchain-google-genai)")

        st.markdown("### 📚 RAG Store")
        if HAS_RAG:
            st.success("RAG store loaded · Marketplace data included")
        else:
            st.warning("RAG disabled")

        st.markdown("### 💾 Storage")
        try:
            mongo_client.admin.command("ping")
            st.success("MongoDB connected")
        except Exception:
            st.error("MongoDB not connected")

    # Navigation
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to:",
        ["New Analysis", "Analysis History"],
        label_visibility="collapsed"
    )

    st.divider()

    # Statistics
    with st.expander("📊 Statistics", expanded=False):
        try:
            total = debates_collection.count_documents({})
            st.metric("Total Analyses", total)
            st.caption("💡 Each analysis includes AI debate + market insights")
        except Exception:
            st.metric("Total Analyses", "—")
            st.caption("Database connection issue")

    # Quick Actions
    with st.expander("⚡ Quick Actions", expanded=False):
        if st.button("🔄 Refresh RAG Data", help="Reload marketplace intelligence"):
            try:
                initialize_marketplace_data()
                st.success("RAG data refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")

        if st.button("📊 View System Info", help="Show technical details"):
            st.info(f"""
            **Technical Details:**
            - Python: 3.13
            - Streamlit: 1.39.0
            - LangChain: v1.x
            - FAISS: Vector store
            - MongoDB: Document storage
            """)



if page == "New Analysis":
    show_new_analysis_page()
else:
    show_history_page()
