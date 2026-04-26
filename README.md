# 🎓 Multi-Agent RAG for Syllabus-Constrained AI Tutoring

An advanced, Multi-Agent Retrieval-Augmented Generation (RAG) architecture designed to strictly enforce pedagogical boundaries and eliminate "Knowledge Bleed" in educational AI systems.

Unlike standard AI chatbots that suffer from conversational amnesia or invent facts outside the syllabus (Epistemic Boundary Violations), this system uses a latency-optimized **Two-Gate Verification Pipeline** and a team of specialized AI agents to guarantee syllabus accuracy.

---

## ✨ Core Research & Architecture Features

- **🛡️ Two-Gate Verification Pipeline:** A novel "early-exit" strategy for LLM reflection. A deterministic fuzzy-matcher (Gate 1) evaluates drafts in milliseconds. Only flagged drafts are passed to the heavy LLM Critic (Gate 2), reducing average inference latency by 27.3% compared to standard iterative reflection (Reflexion) while *increasing* factual faithfulness.
- **🧠 Two-Pass Semantic Chunking:** Replaces arbitrary character-count splitting. It utilizes sentence embeddings and standard-deviation thresholds to identify natural topic boundaries, paired with a strict character cap to preserve complex academic definitions.
- **🤖 Multi-Agent Orchestration:** 
  - **Query Rewriter & Planner:** Fixes conversational amnesia and decomposes complex multi-part queries.
  - **Document Grader (CRAG):** Evaluates retrieved text chunks and discards irrelevant data prior to generation.
  - **Repair Agent:** Iteratively scrubs hallucinated out-of-syllabus facts from the generated draft.
- **⚡ Active Recall Generation:** Instantly generates tailored Multiple Choice Quizzes, Flashcards, and 1-page Summaries.
- **🎙️ Multimodal Voice Input:** Speak naturally to the assistant using integrated real-time voice-to-text functionality.

---

## 📊 Performance Benchmarks
Evaluated via the **Ragas framework** on a curated probing dataset for Data Structures and Algorithms (DSA). Our Proposed Two-Gate system successfully prevented "Reflection Degradation" (over-editing) caused by forced pure-LLM loops, achieving the highest pedagogical safety.

| Architecture | Faithfulness | Answer Relevancy | Context Precision | Context Recall | End-to-End Latency |
|--------------|--------------|------------------|-------------------|----------------|--------------------|
| **Basic RAG** | 0.7875 | 0.9297 | 0.7542 | 0.8000 | ~15.0s |
| **CRAG Baseline** | 0.9487 | 0.9207 | 0.8430 | **0.9500** | 24.2s |
| **Reflexion Baseline** | 0.8988 | 0.8767 | **0.8622** | **0.9500** | 45.4s |
| **Proposed Two-Gate** | **0.9557** | 0.9207 | 0.8469 | **0.9500** | **33.0s** |

*(Note: End-to-End latency includes heavy API network transit and automated rate-limit queuing on the Groq free tier).*

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend UI | Streamlit (Python) |
| Orchestration | LangChain & LangChain Experimental |
| Core LLM | Meta Llama-3.1-8b-instant (via Groq API) |
| Audio Transcription | Whisper-large-v3 (via Groq API) |
| Vector Database | ChromaDB (Persistent local storage) |
| Embedding Model | HuggingFace `all-MiniLM-L6-v2` |

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/GurugubelliAjay/Multi-Agent-RAG-Based-Teaching-Assistant.git
cd Multi-Agent-RAG-Based-Teaching-Assistant
```

### 2. Create and Activate a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Make sure your virtual environment is activated, then run:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your Groq API key (used for lightning-fast Llama-3 inference):
```ini
GROQ_API_KEY=your_groq_api_key_here
```

---

## 💻 Running the Application

To prevent ChromaDB background telemetry errors (`capture() takes 1 positional argument`) and HuggingFace parallelization warnings, start the app with specific environment variables.

### Option A: Using the Launch Script (Recommended)

If you are on a bash terminal (Git Bash, WSL, Mac/Linux):
```bash
chmod +x run.sh
./run.sh
```

### Option B: Manual Command (Windows PowerShell / CMD)
```cmd
set ANONYMIZED_TELEMETRY=False
set TOKENIZERS_PARALLELISM=false
streamlit run web_app.py
```

---

## 📚 How to Use

1. **Create a Course:** Use the sidebar to create a new subject workspace (e.g., "Data Structures").
2. **Upload Materials:** Go to the "Files" section and upload your course PDFs. The Semantic Chunker will process them automatically.
3. **Ask Questions:** Use the Chat tab to type or speak your questions. Watch the Agent Status indicator as it retrieves, grades, and fact-checks its answer.
4. **Generate Study Aids:** Navigate to the Quiz, Summary, or Flashcards tabs to automatically generate active recall materials based on the uploaded PDFs.

---

## 👨‍💻 Project Authors

| Name | Roll Number |
|------|-------------|
| G. Ajay | 1602-22-737-131 |
| N. Prakash | 1602-22-737-158 |
