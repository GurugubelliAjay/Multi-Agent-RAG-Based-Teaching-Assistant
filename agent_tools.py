import os
import sys

# --- CRITICAL CONFIGURATION (MUST BE FIRST) ---
# Disable ChromaDB telemetry to stop "capture()" errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Disable HuggingFace parallelism to stop "fork" warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load secrets immediately
from dotenv import load_dotenv
load_dotenv()

# --- STANDARD IMPORTS ---
import re
import random
import shutil
import json
import sqlite3
import time
import io 
import concurrent.futures

# --- HEAVY IMPORTS (Now safe to load) ---
import chromadb 
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq 
from langchain_experimental.text_splitter import SemanticChunker
# --- NEW IMPORTS FOR ADVANCED AGENTS ---
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import time

# --- DIRECTORY SETUP ---
DB_DIR = "db"
SUBJECTS_DIR = "subjects"
CHAT_DB_FILE = "chat_history.db"
CHROMA_CLIENT = None

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(SUBJECTS_DIR, exist_ok=True)

# --- 1. PERSISTENT CHAT DB ---
def init_chat_db():
    conn = sqlite3.connect(CHAT_DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats 
                 (id INTEGER PRIMARY KEY, subject TEXT, role TEXT, content TEXT)''')
    conn.commit()
    conn.close()

def save_message(subject, role, content):
    conn = sqlite3.connect(CHAT_DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chats (subject, role, content) VALUES (?, ?, ?)", (subject, role, content))
    conn.commit()
    conn.close()

def load_chat_history(subject):
    conn = sqlite3.connect(CHAT_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chats WHERE subject=?", (subject,))
    return [{"role": r[0], "content": r[1]} for r in c.fetchall()]

def clear_chat_history(subject):
    conn = sqlite3.connect(CHAT_DB_FILE)
    c = conn.cursor()
    # Delete all rows matching the subject
    c.execute("DELETE FROM chats WHERE subject=?", (subject,))
    conn.commit()
    conn.close()
    print(f"DEBUG: Database cleared for {subject}") # Useful for your terminal logs

init_chat_db()

# --- 2. CORE TOOLS ---
def get_embeddings(): return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_safe_client(persist_dir):
    # Try up to 3 times to connect, wiping the folder if it fails
    for attempt in range(3):
        try:
            return chromadb.PersistentClient(path=persist_dir)
        except Exception as e:
            if attempt < 2:
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir, ignore_errors=True)
                time.sleep(1) # Wait for OS to release file locks
            else:
                raise e

# In agent_tools.py

def rewrite_query(subject, user_question):
    """
    Rewriter Agent: Contextualizes the user's question based on chat history.
    Example: "Tell me more" -> "Tell me more about the definition of photosynthesis."
    """
    # 1. Get the last few messages for context
    history = load_chat_history(subject)
    
    # If no history, the question is already standalone
    if not history:
        return user_question
        
    # Take the last 2 turns (User + Assistant)
    # We slice [-2:] to get the most immediate context
    recent_history = history[-2:]
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    print(f"--- AGENT: REWRITING QUERY ---\nOriginal: {user_question}")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    system_prompt = """You are a Query Rewriter. 
    Your job is to reformulate the last user question into a standalone search query that can be understood without the chat history.
    
    RULES:
    1. If the question uses pronouns like "it", "that", "he", "they", replace them with the specific nouns from the history.
    2. If the question is already clear (e.g., "What is a cell?"), return it exactly as is.
    3. Do NOT answer the question. Just rewrite it for search.
    4. Output ONLY the rewritten string.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Chat History:\n{history_str}\n\nLatest Question: {user_question}")
    ])
    
    rewriter = prompt | llm
    
    try:
        new_query = rewriter.invoke({}).content.strip()
        print(f"Rewritten: {new_query}")
        return new_query
    except Exception as e:
        print(f"Rewriter Failed: {e}")
        return user_question
    

def grade_documents(question, documents, status_container=None):
    """Evaluator Agent: Batch Grades all documents in a SINGLE API call."""
    if status_container: status_container.write("⚖️ **Grader Agent (CRAG):** Evaluating document relevance in batch mode...")
    print("--- AGENT: GRADING DOCUMENTS (BATCH MODE) ---")
    
    if not documents: 
        if status_container: status_container.write("⚠️ *Filtering:* No documents to grade.")
        return []
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    doc_text = "".join([f"\n--- Document {i} ---\n{d.page_content}\n" for i, d in enumerate(documents)])
        
    system = """You are a strict grader evaluating document relevance.
    Analyze each document and determine if it helps answer the user's question.
    Output ONLY a JSON list of the integers representing the relevant Document IDs.
    If none are relevant, output an empty list []."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}\n\nDocuments:\n{doc_text}")
    ])
    
    try:
        res = (prompt | llm).invoke({"question": question, "doc_text": doc_text}).content
        match = re.search(r'\[.*\]', res, re.DOTALL)
        if match:
            relevant_ids = json.loads(match.group(0))
            filtered_docs = [documents[i] for i in relevant_ids if isinstance(i, int) and i < len(documents)]
            if status_container: status_container.write(f"✅ *Filtering Complete:* Kept **{len(filtered_docs)}/{len(documents)}** relevant chunks.")
            print(f"DEBUG: Batch Grader kept {len(filtered_docs)}/{len(documents)} docs.")
            return filtered_docs
    except Exception as e:
        print(f"Batch Grading Failed: {e}")
        
    if status_container: status_container.write(f"⚠️ *Fallback:* Kept all {len(documents)} docs due to parse error.")
    return documents

def check_hallucinations(context_text, answer):
    """
    Hallucination Grader: Checks if the answer is grounded in the docs.
    """
    print("--- AGENT: CHECKING HALLUCINATIONS ---")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'yes' means the answer is fully supported by the context."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    
    grader = hallucination_prompt | llm
    score = grader.invoke({"documents": context_text, "generation": answer})
    
    return "yes" in score.content.lower()

def planner_agent(question, status_container=None):
    """1. PLANNER AGENT: Breaks complex questions into sub-queries for multi-hop reasoning."""
    if status_container: status_container.write("🧠 **Planner Agent:** Analyzing query for multi-hop reasoning...")
    print("--- AGENT: PLANNING ---")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt = f"""You are a Task Planner. Break this user query into 1 to 3 logical sub-questions for a search engine. 
    Query: {question}
    Output ONLY a valid JSON list of strings. Example: ["What is a stack?", "What is a queue?", "Differences between them?"]"""
    
    try:
        res = llm.invoke(prompt).content
        match = re.search(r'\[.*\]', res, re.DOTALL)
        if match: 
            sub_qs = json.loads(match.group(0))
            if status_container: status_container.write(f"🔀 *Sub-queries generated:* `{sub_qs}`")
            return sub_qs
    except Exception as e:
        print(f"Planner Fallback: {e}")
    
    if status_container: status_container.write(f"🔀 *Using original query:* `['{question}']`")
    return [question]

def hybrid_retrieve(vectorstore, query, k=5):
    """8. MULTI-RETRIEVER: Combines Vector Search (meaning) with BM25 (exact keyword)."""
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    try:
        # Build BM25 keyword index on the fly from current Chroma docs
        db_data = vectorstore.get()
        docs = db_data.get('documents', [])
        metas = db_data.get('metadatas', [])
        
        if docs:
            doc_objs = [Document(page_content=txt, metadata=m) for txt, m in zip(docs, metas)]
            bm25_retriever = BM25Retriever.from_documents(doc_objs)
            bm25_retriever.k = k
            
            # Combine them: 50% Vector weight, 50% Keyword weight
            ensemble = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])
            return ensemble.invoke(query)
    except Exception as e:
        print(f"Hybrid Retrieval Fallback (Using Vector Only): {e}")
    
    return vector_retriever.invoke(query)

def critic_and_repair_agent(context, draft_answer, question, max_loops=2, status_container=None):
    """2 & 3. COLLABORATION & REPAIR: Critic finds errors, Repair fixes them."""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    current_answer = draft_answer
    
    for attempt in range(max_loops):
        if status_container: status_container.write(f"🕵️ **Critic Agent:** Fact-checking draft against source text (Attempt {attempt+1})...")
        print(f"--- AGENT: CRITIC (Attempt {attempt+1}) ---")
        
        critic_prompt = f"""Analyze this answer against the context. 
        Context: {context}
        Question: {question}
        Answer: {current_answer}
        If the answer contains hallucinated info NOT in the context, or misses the core question, output "FAIL: [Reason]".
        If it is perfectly grounded and accurate, output "PASS"."""
        
        critique = llm.invoke(critic_prompt).content
        
        if "PASS" in critique.upper():
            if status_container: status_container.write("🏆 *Validation Pass:* Answer is 100% grounded in syllabus context.")
            print("DEBUG: Critic approved answer.")
            return current_answer, True
            
        if status_container: status_container.write(f"🛠️ **Repair Agent:** Fixing hallucinations detected by Critic...")
        print(f"--- AGENT: REPAIR (Fixing flaws) ---")
        
        repair_prompt = f"""You are a Repair Agent. 
        Context: {context}
        Question: {question}
        Flawed Answer: {current_answer}
        Critic Feedback: {critique}
        Fix the answer based on the feedback. Use ONLY the provided context."""
        
        current_answer = llm.invoke(repair_prompt).content
        
    return current_answer, False

def evaluation_agent(question, answer, relevant_docs, start_time, status_container=None):
    """4 & 10. METRICS & CONFIDENCE: Calculates quantitative system scores."""
    if status_container: status_container.write("📊 **Evaluation Agent:** Calculating confidence and precision metrics...")
    print("--- AGENT: EVALUATION ---")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt = f"""Evaluate this RAG system response.
    Question: {question}
    Answer: {answer}
    Output ONLY a JSON object with these exact keys (values as integers 0-100):
    {{"confidence_score": 90, "accuracy_score": 85, "retrieval_precision": 80}}"""
    
    try:
        res = llm.invoke(prompt).content
        match = re.search(r'\{.*\}', res, re.DOTALL)
        metrics = json.loads(match.group(0))
    except:
        metrics = {"confidence_score": "N/A", "accuracy_score": "N/A", "retrieval_precision": "N/A"}
        
    metrics["response_time"] = round(time.time() - start_time, 2)
    metrics["sources_used"] = len(relevant_docs)
    return metrics

import concurrent.futures

def agentic_rag_response(subject, question, status_container=None):
    """Advanced Orchestrator with UI Status Updates"""
    start_time = time.time()
    
    # 1. PLANNER
    sub_questions = planner_agent(question, status_container)
    
    client = get_shared_client(DB_DIR)
    try:
        vectorstore = Chroma(client=client, collection_name=subject, embedding_function=get_embeddings())
    except:
        return "Error: Could not access subject database."
        
    # 2. RETRIEVE & POOL
    raw_pooled_docs = []
    unique_content = set()
    for sq in sub_questions:
        docs = hybrid_retrieve(vectorstore, sq, k=4) 
        for d in docs:
            if d.page_content not in unique_content:
                unique_content.add(d.page_content)
                raw_pooled_docs.append(d)
                
    if status_container: status_container.write(f"📚 **Hybrid Retriever:** Pooled **{len(raw_pooled_docs)}** unique document chunks from vector & keyword search.")
    
    # 3. BATCH GRADE
    all_relevant_docs = grade_documents(question, raw_pooled_docs, status_container)
                
    if not all_relevant_docs:
        return "I checked your notes, but I couldn't find relevant information. Please try rephrasing."
        
    context_text = "\n\n".join([d.page_content for d in all_relevant_docs])
    unique_sources = set([f"- *{os.path.basename(doc.metadata.get('source', 'Unknown'))}* (Page {doc.metadata.get('page', 0) + 1})" for doc in all_relevant_docs])
    sources_text = "\n\n**📚 Sources Used:**\n" + "\n".join(unique_sources)

    # 4. GENERATE
    if status_container: status_container.write("✍️ **Generator Agent:** Drafting initial response using verified context...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    sys_prompt = f"You are a strict Teaching Assistant for {subject}. Answer ONLY based on the context:\n{context_text}"
    draft_answer = llm.invoke([("system", sys_prompt), ("human", question)]).content
    
    # 5. CRITIC & REPAIR
    final_answer, is_grounded = critic_and_repair_agent(context_text, draft_answer, question, 2, status_container)
    
    # 6. EVALUATE
    metrics = evaluation_agent(question, final_answer, all_relevant_docs, start_time, status_container)
    
    metrics_text = (f"\n\n---\n**📊 Agent Evaluation Metrics:**\n"
                    f"- **Confidence Score:** {metrics.get('confidence_score', 'N/A')}%\n"
                    f"- **Accuracy Score:** {metrics.get('accuracy_score', 'N/A')}%\n"
                    f"- **Retrieval Precision:** {metrics.get('retrieval_precision', 'N/A')}%\n"
                    f"- **Sources Combined:** {metrics.get('sources_used', 0)} document chunks\n"
                    f"- **Response Time:** {metrics.get('response_time', 'N/A')} seconds\n")
    
    if not is_grounded:
        return f"**⚠️ Repair Agent Warning:** Answer failed final hallucination checks but could not be perfectly repaired.\n\n{final_answer}\n{sources_text}{metrics_text}"
        
    return final_answer + "\n" + sources_text + metrics_text
    
def get_rag_chain(subject):
    persist_dir = os.path.join(DB_DIR) # Use the root DB dir
    if not os.path.exists(persist_dir): return None
    
    # Use the shared client instead of creating a new one
    client = get_shared_client(persist_dir)
    
    # Check if collection exists
    try:
        # Check if the subject actually has data in the client
        collections = [c.name for c in client.list_collections()]
        if subject not in collections: return None
        
        vectorstore = Chroma(
            client=client, 
            collection_name=subject, 
            embedding_function=get_embeddings()
        )
    except: return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    system_prompt = (
        f"You are a strict Teaching Assistant for {subject}. "
        "Use ONLY the following context to answer. If not there, say you don't know.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

def get_shared_client(persist_dir):
    global CHROMA_CLIENT
    if CHROMA_CLIENT is None:
        # Use the root DB_DIR so one client manages all subjects
        CHROMA_CLIENT = chromadb.PersistentClient(path=DB_DIR)
    return CHROMA_CLIENT

def handle_file_upload(subject_name, uploaded_files):
    subject_path = os.path.join(SUBJECTS_DIR, subject_name)
    os.makedirs(subject_path, exist_ok=True)
    
    client = get_shared_client(DB_DIR)

    for f in uploaded_files:
        save_path = os.path.join(subject_path, f.name)
        with open(save_path, "wb") as file:
            file.write(f.getbuffer())
        
        loader = PyPDFLoader(save_path)
        docs = loader.load()
        
        if docs:
            print(f"DEBUG: Starting Semantic Chunking for {f.name}...")
            
            # --- THE UPGRADE: SEMANTIC CHUNKER ---
            # Instead of fixed size, we split based on meaning.
            # 'percentile' threshold: strictness of splitting. 
            # 95 means "only split if the topic changes significantly"
            text_splitter = SemanticChunker(
                get_embeddings(), 
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95
            )
            
            splits = text_splitter.split_documents(docs)
            print(f"DEBUG: Created {len(splits)} semantic chunks.")
            # -------------------------------------
            
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=get_embeddings(),
                collection_name=subject_name,
                persist_directory=DB_DIR,
                client=client
            )
            time.sleep(1) 
            print(f"DEBUG: Indexed {len(splits)} chunks into {subject_name}")

def delete_file(subject, filename):
    path = os.path.join(SUBJECTS_DIR, subject, filename)
    if os.path.exists(path): os.remove(path)
    handle_file_upload(subject, []) # Rebuild empty or remaining

def delete_subject(subject):
    s_path = os.path.join(SUBJECTS_DIR, subject)
    d_path = os.path.join(DB_DIR, subject)
    if os.path.exists(s_path): shutil.rmtree(s_path)
    if os.path.exists(d_path): shutil.rmtree(d_path)
    clear_chat_history(subject)

def list_files(subject):
    path = os.path.join(SUBJECTS_DIR, subject)
    return [f for f in os.listdir(path) if f.endswith(".pdf")] if os.path.exists(path) else []

def generate_quiz(subject):
    client = get_shared_client(DB_DIR)
    try:
        vectorstore = Chroma(client=client, collection_name=subject, embedding_function=get_embeddings())
        
        # 1. SEARCH ANGLES (To prevent repetitive questions)
        search_angles = [
            "core definitions and key concepts",
            "historical context and background",
            "mathematical formulas and theories",
            "limitations and critical analysis",
            "real-world applications",
            "comparison of major topics"
        ]
        selected_angle = random.choice(search_angles)
        print(f"DEBUG: Quiz focusing on '{selected_angle}'")
        
        # 2. RETRIEVAL
        retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(selected_angle)
        
        if not docs:
            # Fallback: Grab Intro/Outro chunks if search fails
            all_data = vectorstore.get()
            if all_data and all_data['documents']:
                docs_list = all_data['documents']
                indices = list(range(min(3, len(docs_list)))) + list(range(max(0, len(docs_list)-3), len(docs_list)))
                selected_docs = [docs_list[i] for i in set(indices)]
                context = "\n".join(selected_docs)
            else:
                return None
        else:
            # Randomly pick 5 chunks from the search results
            selected_docs = random.sample(docs, min(len(docs), 5))
            context = "\n".join([d.page_content for d in selected_docs])
            
    except Exception as e:
        print(f"Quiz Database Error: {e}")
        return None

    # 3. GENERATION
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
    
    # We explicitly ask for EXACTLY 3 questions
    prompt = f"""
    You are an expert exam setter. 
    Based on the text below, generate EXACTLY 3 unique Multiple Choice Questions.
    
    FOCUS AREA: {selected_angle.upper()}
    TEXT CONTEXT: "{context[:5000]}"
    
    CRITICAL INSTRUCTIONS:
    - Return ONLY a valid JSON array.
    - Format: [{{"question": "...", "options": ["A","B","C","D"], "answer": "Exact Option Text", "explanation": "..."}}]
    """
    
    try:
        res = llm.invoke(prompt).content
        
        # --- THE FIX: raw_decode ---
        # This ignores "Extra data" errors by stopping at the end of the JSON list.
        
        # 1. Clean Markdown wrappers
        clean_res = res.replace("```json", "").replace("```", "").strip()
        
        # 2. Find the start of the list
        start_idx = clean_res.find("[")
        if start_idx == -1: return []
        
        # 3. Decode only the valid JSON part
        # raw_decode returns (parsed_object, end_index)
        # We only care about the parsed_object
        parsed_json, _ = json.JSONDecoder().raw_decode(clean_res[start_idx:])
        
        return parsed_json
            
    except Exception as e: 
        print(f"Quiz JSON Error: {e} | Raw: {res[:50]}...")
        return []

def generate_summary(subject):
    client = get_shared_client(DB_DIR)
    try:
        collections = [c.name for c in client.list_collections()]
        if subject not in collections: return "No documents found."
        
        vectorstore = Chroma(client=client, collection_name=subject, embedding_function=get_embeddings())
        # Increase k to get a broader view of the document for a summary
        docs = vectorstore.as_retriever(search_kwargs={"k": 15}).invoke("main topics and overview")
        context = "\n".join([d.page_content for d in docs])
    except: return "Error accessing documents."
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    return llm.invoke(f"Create a 1-page Cheat Sheet from:\n{context[:12000]}").content

def generate_flashcards(subject):
    client = get_shared_client(DB_DIR)
    try:
        vectorstore = Chroma(client=client, collection_name=subject, embedding_function=get_embeddings())
        
        # 1. SEARCH FOR DENSE CONTENT
        # We look for terms that likely contain lists, definitions, and examples
        search_query = "key concepts definitions examples characteristics types"
        
        # Fetch more context (k=15) to ensure we have enough detail
        docs = vectorstore.as_retriever(search_kwargs={"k": 15}).invoke(search_query)
        
        if not docs:
            # Fallback: If search fails, grab random chunks
            all_data = vectorstore.get()
            if all_data and all_data['documents']:
                docs_list = all_data['documents']
                sample_size = min(len(docs_list), 5)
                context = "\n".join(random.sample(docs_list, sample_size))
            else:
                return None
        else:
            # Shuffle the results to get different cards every time
            selected_docs = random.sample(docs, min(len(docs), 5))
            context = "\n".join([d.page_content for d in selected_docs])
            
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return None

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
    
    # 2. THE RICH PROMPT
    # We explicitly ask for "Comprehensive details"
    prompt = f"""
    Create 5 DETAILED flashcards from this text.
    
    TEXT CONTEXT:
    "{context[:6000]}"
    
    CRITICAL INSTRUCTIONS:
    - The "back" of the card must contain MORE than just a definition.
    - Include: Definition, Key Characteristics, and an Example if possible.
    - Output ONLY a valid JSON array.
    - Format: [{{"front": "Term", "back": "**Definition:** ...\\n\\n**Key Points:** ...\\n\\n**Example:** ..."}}]
    """
    
    try:
        res = llm.invoke(prompt).content
        
        # 3. ROBUST PARSING (The Regex Fix)
        # This regex looks for a pattern that starts with [ and ends with ]
        # It handles newlines and nested braces correctly
        match = re.search(r'\[.*\]', res, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            print(f"DEBUG: No JSON array found in response: {res[:100]}")
            return []
            
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        return []
    except Exception as e:
        print(f"General Error: {e}")
        return []

# --- 5. NEW: AUDIO TRANSCRIPTION ---
def transcribe_audio(audio_bytes):
    """Sends audio bytes to Groq Whisper for transcription."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # Create a file-like object
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav" # Groq needs a filename
    
    try:
        transcription = client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
        return transcription
    except Exception as e:
        return f"Error transcribing: {str(e)}"
    

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

def create_pdf_bytes(text, title="Subject Summary"):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, title)
    
    # Content
    p.setFont("Helvetica", 12)
    y_position = height - 80
    
    # Split text to fit page width
    lines = text.split('\n')
    for line in lines:
        # Wrap long lines
        wrapped_lines = simpleSplit(line, "Helvetica", 12, width - 100)
        for w_line in wrapped_lines:
            if y_position < 50:  # Start a new page if we run out of space
                p.showPage()
                p.setFont("Helvetica", 12)
                y_position = height - 50
            p.drawString(50, y_position, w_line)
            y_position -= 15
        y_position -= 5 # Extra space between paragraphs
        
    p.save()
    buffer.seek(0)
    return buffer