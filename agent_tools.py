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
    

def grade_documents(question, documents):
    """
    Evaluator Agent: Checks if retrieved docs are relevant.
    Returns: A list of filtered, relevant documents.
    """
    print("--- AGENT: GRADING DOCUMENTS ---")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Prompt to force binary JSON decision
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    grader_llm = grade_prompt | llm
    
    filtered_docs = []
    for d in documents:
        try:
            score = grader_llm.invoke({"question": question, "document": d.page_content})
            # Simple parsing check
            if "yes" in score.content.lower():
                print(f"DEBUG: Document kept (Relevant)")
                filtered_docs.append(d)
            else:
                print(f"DEBUG: Document filtered out (Irrelevant)")
        except:
            filtered_docs.append(d) # Keep if grading fails to be safe
            
    return filtered_docs

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

def agentic_rag_response(subject, question):
    """
    Orchestrator: REWRITE -> Retrieval -> Grading -> Generation -> Verification -> CITATION
    """
    # --- STEP 0: REWRITE QUERY ---
    # We use the rewritten query for searching, but the original question for generating the answer
    # (keeps the conversation natural).
    search_query = rewrite_query(subject, question)
    
    # 1. RETRIEVAL (Use search_query, NOT question)
    client = get_shared_client(DB_DIR)
    try:
        vectorstore = Chroma(client=client, collection_name=subject, embedding_function=get_embeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        documents = retriever.invoke(search_query) # <--- UPDATED
    except:
        return "Error: Could not access subject database."

    # 2. CRAG (Corrective RAG) - Filter Docs
    # We check relevance against the CLEARED UP query
    relevant_docs = grade_documents(search_query, documents) # <--- UPDATED
    
    # 3. FALLBACK MECHANISM
    if not relevant_docs:
        return "I checked your notes, but I couldn't find any information relevant to that specific question. Please try rephrasing or check the uploaded PDFs."
    
    # --- CITATION LOGIC ---
    unique_sources = set()
    for doc in relevant_docs:
        file_path = doc.metadata.get("source", "Unknown")
        file_name = os.path.basename(file_path)
        page_num = doc.metadata.get("page", 0) + 1 
        unique_sources.add(f"- *{file_name}* (Page {page_num})")
    sources_text = "\n\n---\n**📚 Sources:**\n" + "\n".join(sorted(unique_sources))

    # 4. GENERATION
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    context_text = "\n\n".join([d.page_content for d in relevant_docs])
    
    system_prompt = f"You are a Teaching Assistant for {subject}. Answer the question based ONLY on the following context:\n{context_text}"
    messages = [("system", system_prompt), ("human", question)] # Keep original question for tone
    
    generation = llm.invoke(messages).content
    
    # 5. HALLUCINATION CHECK
    is_grounded = check_hallucinations(context_text, generation)
    
    if is_grounded:
        return generation + sources_text
    else:
        return f"**Warning (Agent Self-Correction):** I generated an answer, but upon reflection, it might not be fully supported by your notes. \n\n{generation}\n{sources_text}"
    
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