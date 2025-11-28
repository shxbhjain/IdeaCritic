# IdeaCritic — LLM-Powered Idea Evaluation & Critique Engine

A structured, multi-stage evaluation system that uses LLM reasoning + RAG + schema-constrained workflows to critique ideas with consistency, depth, and zero hallucination drift.

---

## 🚀 Overview

IdeaCritic is an **LLM-based agent** that analyzes, scores, and critiques user ideas across multiple dimensions (feasibility, clarity, technical complexity, market potential, risks, etc.). It uses a **multi-step reasoning pipeline**, embeddings-driven RAG, and schema-based output constraints to ensure consistent, grounded evaluations.

This project powers an interactive evaluation UI built in Streamlit.

---

## 📌 Features

* Multi-stage idea evaluation workflow (reasoning → scoring → critique → summary).
* RAG pipeline using embeddings + vector search (FAISS or similar).
* Schema-constrained generation to eliminate hallucinated fields.
* Deterministic evaluation outputs (fixed JSON schema per stage).
* Streamlit UI for real-time feedback and refinement.
* Modular architecture for plugging in different LLMs.

---

## 🛠 Tech Stack

* **Backend / Core Logic:** Python
* **LLM Integration:** Gemini API (or any LLM provider)
* **Embeddings + Search:** FAISS / sentence transformers
* **RAG:** Custom context retrieval pipeline
* **UI:** Streamlit
* **Storage:** Local vector DB / FAISS index

---

## 🧩 Architecture Overview

1. **User submits an idea** through UI.
2. System embeds the idea → performs vector search → retrieves relevant context.
3. Multi-step workflow triggers:

   * Stage 1: Idea breakdown & interpretation
   * Stage 2: Dimensional scoring (clarity, feasibility, originality, impact...)
   * Stage 3: Risk analysis
   * Stage 4: Final critique & summary
4. Outputs follow strict JSON schemas for consistency.
5. Streamlit displays structured results + improvement suggestions.

---

## 📁 Project Structure

```
ideacritic/
│── core/
│   ├── evaluator.py
│   ├── workflow.py
│   ├── schemas.py
│   ├── retriever.py
│   └── embeddings.py
│
│── ui/
│   └── app.py (Streamlit)
│
│── data/
│   └── vector_index.faiss
│
│── config/
│   └── settings.py
│
└── README.md
```

---

## ⚙️ Getting Started (Local)

### 1. Clone repo

```
git clone https://github.com/shxbhjain/IdeaCritic.git
cd IdeaCritic
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Set up environment

Create `.env`:

```
GEMINI_API_KEY=your_key_here
VECTOR_DB_PATH=./data/vector_index.faiss
```

### 4. Build vector index (if not included)

```
python core/embeddings.py
```

### 5. Run Streamlit UI

```
streamlit run ui/app.py
```

Open: `http://localhost:8501`

---

## 📡 RAG Pipeline

IdeaCritic uses:

* Embedding model → converts idea text into vectors.
* FAISS index → performs similarity search.
* Retriever → returns top-k documents.

This context is injected into the LLM workflow to ground critiques.

---

## 🔍 Evaluation Workflow

Each stage has a strict JSON schema. Example:

### **Stage: Dimensional Scoring**

```
{
  "clarity": 0–10,
  "feasibility": 0–10,
  "technical_complexity": 0–10,
  "originality": 0–10,
  "market_viability": 0–10,
  "explanation": "string"
}
```

### **Stage: Final Critique**

```
{
  "strengths": [...],
  "weaknesses": [...],
  "risks": [...],
  "summary": "string"
}
```

Workflows are chained → outputs of one stage feed into the next.

---

## 🖥 Streamlit UI

The interface allows:

* Input box for idea text
* Buttons to trigger evaluation
* Real-time scoring visualization
* Side-by-side critique + improvement suggestions
* Downloadable evaluation JSON

---

## ☁ Deployment

### **Local / Development**

* Runs on Streamlit directly.

### **Production**

Options:

* Streamlit Cloud
* Docker + VPS
* Railway / Render

Add environment variables in platform settings.

---

## 🧪 Testing

* Unit tests for:

  * Retriever
  * Embedding pipeline
  * Workflow chaining
  * Schema validation

* Manual testing for Streamlit UI

---

## 🛠 Common Issues

**Hallucination / inconsistent JSON:**
Fix by tightening schema in `schemas.py` and using structured prompting.

**RAG not returning relevant results:**
Increase vector dimensions, improve embedding model.

**Index not loading:**
Ensure FAISS is compiled for your platform.

---

## 📈 Roadmap

* Add agentic self-refinement loop
* Add scoring profiles (startup ideas, tech ideas, research ideas)
* Add export formats (PDF/Markdown)
* Add multi-user session support

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `feat/<your-feature>`
3. Commit and push
4. Open PR

---

## 📄 License & Contact

* License: MIT
* Author: Shubh Jain

---

# Quick Copy Header

```
# IdeaCritic — LLM-Powered Idea Evaluation Engine
Structured, RAG-enhanced, schema-based reasoning for consistent idea critique.
```
