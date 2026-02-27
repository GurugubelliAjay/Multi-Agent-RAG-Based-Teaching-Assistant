# 🎓 AI-Powered Multi-Agent Teaching Assistant Using RAG

An advanced, Agentic Retrieval-Augmented Generation (RAG) platform that transforms static academic PDFs into an interactive, self-verifying, and hallucination-free study companion.

Unlike standard AI chatbots that suffer from "amnesia" or invent facts outside the syllabus, this system uses a team of specialized AI agents to grade retrieved documents, rewrite vague queries, and cross-reference answers against your specific course materials.

---

## ✨ Core Features

- **🧠 Semantic Chunking:** Replaces "dumb" character-count splitting. It reads PDFs sentence by sentence and splits text based on meaning (using a 95th-percentile cosine distance threshold), ensuring definitions and complex concepts stay perfectly intact.
- **🛡️ Zero-Hallucination Multi-Agent Workflow:**
  - **Query Rewriter Agent:** Fixes vague follow-up questions using chat history (e.g., changes "What are its advantages?" to "What are the advantages of a Stack?").
  - **Document Grader Agent (CRAG):** Evaluates retrieved text chunks and throws away irrelevant data before the AI reads it.
  - **Hallucination Checker Agent:** Acts as the final boss, cross-referencing the drafted answer against the textbook to guarantee 100% syllabus accuracy.
- **⚡ Active Recall Generation:** Instantly generates tailored Multiple Choice Quizzes, Flashcards, and 1-page Summaries using robust JSON parsing (`raw_decode`) and "Jittered Retrieval" to prevent repetitive questions.
- **🎙️ Multimodal Voice Input:** Speak naturally to the assistant using integrated real-time voice-to-text functionality.

---

## 🛠️ Tech Stack

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

Create a `.env` file in the root directory of the project and add your Groq API key:
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
