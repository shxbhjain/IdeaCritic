# 🚀 IdeaCritic — LLM-Powered Idea Evaluation Engine

**IdeaCritic** is a multi-agent LLM system designed to evaluate early-stage startup ideas through a simulated pitch-panel workflow.

Instead of relying on a single LLM response, IdeaCritic coordinates three specialized AI agents — **Optimist, Critic, and Evaluator** — to analyze an idea from different perspectives and produce a structured final assessment.

The system combines **LLM reasoning, multi-agent orchestration, RAG-based contextual retrieval, FAISS semantic search, prompt engineering, and structured JSON outputs**.

---

## 🎯 Overview

When a user submits a startup idea, IdeaCritic evaluates it through three different perspectives:

* 🟢 **Optimist Agent** — identifies strengths, opportunities, innovation, growth potential, and positive market possibilities.
* 🔴 **Critic Agent** — challenges the idea, identifies weaknesses, feasibility problems, operational risks, and hidden assumptions.
* 🔵 **Evaluator Agent** — acts as the final decision-maker, synthesizes the available analysis and produces scores, risks, recommendations, and a final verdict.

The system ultimately classifies the idea as:

```text
PROCEED
PIVOT
DROP
```

---

# 💡 Problem Statement

Early-stage founders often need fast and unbiased feedback before investing significant time and resources into an idea.

Traditional feedback sources such as mentors, consultants, and pitch panels can be:

* Expensive
* Time-consuming
* Difficult to access
* Subjective

A simple single-prompt LLM evaluation also has limitations. It can produce generic, overly positive, or inconsistent feedback.

### IdeaCritic's approach

Instead of asking:

> "Is this startup idea good?"

IdeaCritic simulates a structured pitch discussion:

```text
Startup Idea
     │
     ▼
  Optimist
     │
     ▼
   Critic
     │
     ▼
  Evaluator
     │
     ▼
Final Decision
```

This forces the system to consider both the **potential upside and potential failure points** before reaching a conclusion.

---

# 🤖 Multi-Agent Architecture

## 1. Optimist Agent 🟢

The Optimist analyzes the idea from a positive perspective.

It focuses on:

* Strengths
* Market opportunities
* Innovation
* Growth potential
* Positive use cases
* Potential customer value

Example:

```text
Idea:
AI voice agent for appointment booking.

Optimist:
- Reduces repetitive receptionist work
- Provides 24/7 availability
- Can potentially scale across clinics
- Addresses a clear operational problem
```

The goal is to build the **strongest reasonable case for the idea**.

---

## 2. Critic Agent 🔴

The Critic acts as a devil's advocate.

It challenges assumptions made by the idea and the Optimist.

It analyzes:

* Feasibility
* Operational risks
* Technical limitations
* Competition
* Scalability
* Hidden assumptions
* Potential failure points

Example:

```text
Critic:
- Voice recognition may introduce booking errors
- Integration with existing systems may be difficult
- Customers may not trust automated calls
- Cost per interaction could become significant
```

The Critic prevents the evaluation from becoming blindly optimistic.

---

## 3. Evaluator Agent 🔵

The Evaluator acts as the final decision-maker.

It receives:

* Original startup idea
* Optimist analysis
* Critic analysis
* Relevant retrieved context

It then synthesizes the information and generates the final evaluation.

### Evaluation dimensions

```text
Clarity
Feasibility
Impact
Market Viability
```

Each dimension receives a score between:

```text
0 – 10
```

The Evaluator also produces:

* Strengths
* Weaknesses
* Risks
* Recommendations
* Final verdict

### Final Verdict

```text
PROCEED
PIVOT
DROP
```

---

# 🔄 End-to-End Pipeline

The complete workflow is:

```text
                 USER
                  │
                  ▼
          ┌───────────────┐
          │   Streamlit   │
          │      UI       │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │   User Idea   │
          │   Processing  │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │   Embedding   │
          │    Model      │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │     FAISS     │
          │ Semantic      │
          │   Search      │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ Retrieved     │
          │ Context       │
          └───────┬───────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ Multi-Agent Workflow│
        └──────────┬──────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
     ┌──────────┐       ┌──────────┐
     │ Optimist │       │  Critic  │
     └────┬─────┘       └────┬─────┘
          │                  │
          └────────┬─────────┘
                   ▼
             ┌───────────┐
             │ Evaluator │
             └─────┬─────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Structured JSON │
          │   Validation    │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Final Evaluation│
          └─────────────────┘
```

---

# 🔍 RAG Pipeline

IdeaCritic uses Retrieval-Augmented Generation to provide relevant contextual information to the LLM evaluation process.

The retrieval pipeline follows:

```text
Reference Information
        │
        ▼
    Embeddings
        │
        ▼
   Vector Index
        │
        ▼
       FAISS
        │
        ▼
 Semantic Similarity Search
        │
        ▼
     Top-K Context
        │
        ▼
      LLM Agents
```

### Why RAG?

Without retrieval, the LLM relies primarily on its existing model knowledge.

With RAG:

```text
User Idea
    +
Retrieved Context
    ↓
LLM Evaluation
```

The retrieved context provides additional information relevant to the idea being evaluated.

RAG is intended to **improve grounding and relevance**. It does not guarantee that the LLM will never hallucinate.

---

# 🧠 Embeddings

Embeddings convert text into numerical vectors representing semantic meaning.

For example:

```text
"AI chatbot for restaurants"
            ↓
       Embedding Model
            ↓
[0.12, -0.43, 0.71, ...]
```

The vector is then searched against the FAISS index to identify semantically similar information.

---

# ⚡ FAISS

FAISS is used for vector similarity search.

Instead of comparing the query against every piece of text manually, IdeaCritic performs vector search to retrieve relevant context.

```text
Query Vector
     │
     ▼
FAISS Index
     │
     ▼
Nearest Vectors
     │
     ▼
Relevant Documents
```

The retrieved information is then supplied to the LLM workflow.

---

# 🧩 Prompt Engineering

Each agent receives a specialized prompt based on its role.

### Optimist

```text
You are an optimistic startup analyst.

Analyze the idea from the perspective
of strengths, opportunities, innovation,
growth potential and market potential.

Identify the strongest reasonable case
for the proposed idea.
```

### Critic

```text
You are a skeptical startup analyst.

Challenge the proposed idea.

Identify:
- Feasibility problems
- Operational risks
- Hidden assumptions
- Technical limitations
- Competitive threats
- Scalability issues

Do not blindly agree with the idea.
```

### Evaluator

```text
You are the final startup evaluator.

Review:
- The original idea
- Optimist analysis
- Critic analysis
- Retrieved context

Produce:
- Dimensional scores
- Risks
- Recommendations
- Final verdict

The verdict must be:
PROCEED, PIVOT, or DROP.
```

This role-based prompting ensures that the same underlying LLM can perform different specialized tasks.

---

# 📦 Structured Output

Each stage produces structured information rather than unrestricted natural-language responses.

Example evaluator output:

```json
{
  "clarity": 8,
  "feasibility": 7,
  "impact": 8,
  "market_viability": 6,
  "verdict": "PIVOT",
  "strengths": [
    "Clear customer problem",
    "Potential for automation"
  ],
  "risks": [
    "Strong competition",
    "Integration complexity"
  ],
  "recommendations": [
    "Focus on a specific customer segment",
    "Validate willingness to pay"
  ]
}
```

Structured outputs make the workflow easier to:

* Validate
* Process
* Display
* Debug
* Pass between stages

Schema validation controls the **structure** of the response; it does not guarantee that every claim generated by the LLM is factually correct.

---

# 🛠 Tech Stack

| Component            | Technology                    |
| -------------------- | ----------------------------- |
| Programming Language | Python                        |
| LLM                  | Gemini API                    |
| AI Architecture      | Multi-Agent Workflow          |
| Retrieval            | RAG                           |
| Vector Search        | FAISS                         |
| Embeddings           | Embedding Model               |
| Prompting            | Role-Based Prompt Engineering |
| Structured Output    | JSON / Schema Validation      |
| Frontend             | Streamlit                     |
| Vector Storage       | Local FAISS Index             |

---

# 📁 Project Structure

```text
ideacritic/
│
├── core/
│   ├── evaluator.py
│   ├── workflow.py
│   ├── schemas.py
│   ├── retriever.py
│   └── embeddings.py
│
├── ui/
│   └── app.py
│
├── data/
│   └── vector_index.faiss
│
├── config/
│   └── settings.py
│
├── README.md
└── requirements.txt
```

### Core Components

#### `workflow.py`

Responsible for orchestrating the multi-agent evaluation workflow.

```text
Idea
 ↓
Optimist
 ↓
Critic
 ↓
Evaluator
 ↓
Final Result
```

#### `evaluator.py`

Handles the LLM-based evaluation logic and final assessment.

#### `schemas.py`

Defines expected structures for agent outputs.

#### `retriever.py`

Handles retrieval of relevant contextual information from the vector index.

#### `embeddings.py`

Responsible for generating embeddings and building/updating the FAISS index.

#### `app.py`

Provides the Streamlit interface.

---

# 🖥 Streamlit Interface

The Streamlit interface allows users to:

1. Enter a startup idea.
2. Submit the idea for evaluation.
3. Run the multi-agent workflow.
4. View Optimist analysis.
5. View Critic analysis.
6. View final Evaluator results.
7. View dimensional scores.
8. View risks and recommendations.
9. View the final Proceed/Pivot/Drop verdict.
10. Export structured evaluation results.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/shxbhjain/IdeaCritic.git
cd IdeaCritic
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
VECTOR_DB_PATH=./data/vector_index.faiss
```

## 4. Build the vector index

```bash
python core/embeddings.py
```

## 5. Run the application

```bash
streamlit run ui/app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🧪 Example Evaluation

### Input

```text
An AI voice agent that handles appointment
booking for small clinics.
```

### Optimist

```text
Strengths:
- Reduces administrative workload
- 24/7 availability
- Potentially scalable
- Clear business use case
```

### Critic

```text
Risks:
- Speech recognition errors
- Integration challenges
- Customer trust
- Potential operational costs
```

### Evaluator

```json
{
  "clarity": 9,
  "feasibility": 7,
  "impact": 8,
  "market_viability": 7,
  "verdict": "PROCEED"
}
```

### Recommendation

```text
Start with a narrow clinic segment and validate
the booking workflow before expanding.
```

---

# 🔐 Reliability Approach

IdeaCritic focuses on improving consistency through multiple mechanisms:

### 1. Role separation

Different agents are responsible for different perspectives.

### 2. Context retrieval

Relevant information can be retrieved through the RAG pipeline.

### 3. Structured outputs

Each stage follows an expected output structure.

### 4. Multi-stage workflow

The evaluation is decomposed instead of relying on one large prompt.

### 5. Validation

Generated outputs can be validated before being passed to subsequent stages.

> These techniques improve consistency and grounding, but they do not guarantee zero hallucinations or perfect evaluation accuracy.

---

# 🚧 Limitations

The system still depends on the underlying LLM and retrieved context.

Potential limitations include:

* Incorrect LLM reasoning
* Irrelevant retrieval results
* Incomplete contextual data
* API failures
* Latency from multiple LLM calls
* Increased token usage from multi-agent workflows
* Subjectivity in startup evaluation

The three-agent architecture improves perspective diversity, but it does not automatically make the final decision objectively correct.

---

# 📈 Future Improvements

Potential improvements include:

* Agent self-reflection and refinement
* Better retrieval evaluation
* Automated retrieval-quality metrics
* More specialized evaluation profiles
* Startup-specific knowledge bases
* Persistent user sessions
* Production vector database
* LLM response caching
* API retry and timeout handling
* Evaluation benchmark datasets
* Human feedback loops
* Cost and latency optimization
* Production monitoring and logging

---

# 🎯 Key Learning Outcomes

Through IdeaCritic, I worked with:

* LLM API integration
* Multi-agent workflow design
* Role-based prompt engineering
* RAG
* Embeddings
* FAISS vector search
* Semantic retrieval
* Structured JSON outputs
* LLM evaluation workflows
* Streamlit application development
* AI response validation

The project helped me understand that building an LLM application involves more than simply calling an API. The surrounding workflow — **prompt design, context retrieval, agent orchestration, structured outputs, validation and quality control** — is equally important.

---

# 👨‍💻 Author

**Shubh Jain**

B.E. Computer Science (Artificial Intelligence)
Chitkara University

GitHub: `https://github.com/shxbhjain`

---

# 📄 License

This project is released under the MIT License.
