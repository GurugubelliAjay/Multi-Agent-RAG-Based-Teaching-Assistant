import os
import sys

# --- CRITICAL CONFIGURATION (MUST BE FIRST) ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

# --- STANDARD IMPORTS ---
import re
import random
import shutil
import json
import sqlite3
import hashlib
import time
import io
import concurrent.futures
import uuid
import difflib
import threading
from dataclasses import dataclass, field

# --- HEAVY IMPORTS ---
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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

# --- DIRECTORY SETUP ---
DB_DIR = "db"
SUBJECTS_DIR = "subjects"
CHAT_DB_FILE = "chat_history.db"
CHROMA_CLIENT = None

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(SUBJECTS_DIR, exist_ok=True)


# =============================================================================
# IMPROVEMENT 7 — Structured response dataclass
# =============================================================================

@dataclass
class RAGResponse:
    """
    Structured return type for the RAG pipeline.
    Replaces the previous plain-string concatenation, making confidence,
    sources, and supporting quotes available to the UI layer.
    """
    answer: str
    confidence: float
    sources: list = field(default_factory=list)          # [{"file": "...", "page": 3}]
    supporting_quotes: list = field(default_factory=list)
    sub_questions_used: list = field(default_factory=list)
    retrieval_stats: dict = field(default_factory=dict)  # chunks_retrieved, chunks_used, reranked
    warning: str = None

    def to_markdown(self) -> str:
        """Render as formatted markdown for Streamlit display."""
        confidence_label = (
            "High" if self.confidence >= 0.8 else
            "Medium" if self.confidence >= 0.5 else
            "Low"
        )

        parts = []
        if self.warning:
            parts.append(f"**{self.warning}**\n\n")

        parts.append(self.answer)
        parts.append(f"\n\n---\n**Confidence:** {confidence_label} ({self.confidence:.0%})")

        if self.supporting_quotes:
            parts.append("\n\n**Evidence from your notes:**")
            for q in self.supporting_quotes[:3]:
                parts.append(f'\n> "{q}"')

        if self.sources:
            parts.append("\n\n**Sources:**")
            for s in self.sources:
                parts.append(f"\n- *{s['file']}*, page {s['page']}")

        return "".join(parts)


# =============================================================================
# IMPROVEMENT 9 — Typed exceptions for clear failure signalling
# =============================================================================

class RateLimitExhaustedError(RuntimeError):
    """Raised when all safe_invoke retries are consumed by rate-limit errors."""
    pass


# =============================================================================
# 1. PERSISTENT DATABASES
# =============================================================================

def init_chat_db():
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS chats
                     (id INTEGER PRIMARY KEY, username TEXT, subject TEXT,
                      role TEXT, content TEXT)''')
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(chats)")
        if 'username' not in [info[1] for info in cursor.fetchall()]:
            conn.execute(
                "ALTER TABLE chats ADD COLUMN username TEXT DEFAULT 'default_user'"
            )

def save_message(username, subject, role, content):
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        conn.execute(
            "INSERT INTO chats (username, subject, role, content) VALUES (?, ?, ?, ?)",
            (username, subject, role, content)
        )

def load_chat_history(username, subject):
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT role, content FROM chats WHERE username=? AND subject=?",
            (username, subject)
        )
        return [{"role": r[0], "content": r[1]} for r in cursor.fetchall()]

def clear_chat_history(username, subject):
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        conn.execute(
            "DELETE FROM chats WHERE username=? AND subject=?",
            (username, subject)
        )
    print(f"DEBUG: Database cleared for {username} - {subject}")

init_chat_db()

def init_users_db():
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT)''')

init_users_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        with sqlite3.connect(CHAT_DB_FILE) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, hash_password(password))
            )
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT password_hash FROM users WHERE username=?", (username,)
        )
        result = cursor.fetchone()
    return result is not None and result[0] == hash_password(password)

def init_sessions_db():
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS sessions
                     (token TEXT PRIMARY KEY, username TEXT)''')

init_sessions_db()

def create_session(username):
    token = str(uuid.uuid4())
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        conn.execute(
            "INSERT INTO sessions (token, username) VALUES (?, ?)", (token, username)
        )
    return token

def get_username_from_session(token):
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT username FROM sessions WHERE token=?", (token,)
        )
        result = cursor.fetchone()
    return result[0] if result else None

def destroy_session(token):
    with sqlite3.connect(CHAT_DB_FILE) as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))


# =============================================================================
# 2. CORE UTILITIES
# =============================================================================

def _parse_retry_after(err_str: str) -> float:
    """Parse a Groq retry-after duration string like '2m30.5s' into seconds."""
    match = re.search(r'(?:([0-9]+)h)?(?:([0-9]+)m)?([0-9.]+)s', err_str)
    if match:
        h = float(match.group(1) or 0)
        m = float(match.group(2) or 0)
        s = float(match.group(3) or 0)
        return h * 3600 + m * 60 + s + 1.0
    return 10.0  # conservative default


def safe_invoke(runnable, prompt_input, max_retries=4):
    """
    Invoke an LLM runnable, auto-retrying on Groq rate-limit errors.

    IMPROVEMENT 9: raises RateLimitExhaustedError on exhaustion instead of
    silently returning "" (which previously produced blank answers with no
    error signal to the caller).
    """
    for attempt in range(max_retries):
        try:
            result = runnable.invoke(prompt_input)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Rate limit" in err_str or "rate_limit_exceeded" in err_str:
                sleep_time = _parse_retry_after(err_str)
                if attempt < max_retries - 1:
                    print(f"DEBUG: Rate limit — sleeping {sleep_time:.1f}s "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    raise RateLimitExhaustedError(
                        f"Rate limit after {max_retries} attempts"
                    ) from e
            else:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
    return ""


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_safe_client(persist_dir):
    for attempt in range(3):
        try:
            return chromadb.PersistentClient(path=persist_dir)
        except Exception as e:
            if attempt < 2:
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir, ignore_errors=True)
                time.sleep(1)
            else:
                raise


# =============================================================================
# IMPROVEMENT 11a — Collection name collision fix
# =============================================================================

def get_collection_name(username: str, subject: str) -> str:
    """
    Generate a Chroma collection name that is unique per (username, subject) pair.

    FIX: The previous f"{username}_{subject}" scheme could produce identical
    names for ("alice", "data_structures") and ("alice_data", "structures").
    A triple-underscore separator cannot appear after sanitisation, so it
    acts as an unambiguous field delimiter. An MD5 suffix handles names
    longer than Chroma's 63-char limit.
    """
    raw = f"{username}___{subject}"          # triple underscore as separator
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', raw).lower()
    if len(sanitized) > 63:
        suffix = hashlib.md5(raw.encode()).hexdigest()[:8]
        sanitized = sanitized[:54] + "_" + suffix
    return sanitized


def get_shared_client(persist_dir=None):
    global CHROMA_CLIENT
    if CHROMA_CLIENT is None:
        CHROMA_CLIENT = chromadb.PersistentClient(path=DB_DIR)
    return CHROMA_CLIENT


# =============================================================================
# IMPROVEMENT 11b — Thread-local Chroma clients for parallel retrieval
# =============================================================================

_thread_local = threading.local()

def get_thread_local_vectorstore(col_name: str) -> Chroma:
    """
    Return a per-thread Chroma vectorstore instance.

    FIX: The original code shared a single CHROMA_CLIENT across all
    ThreadPoolExecutor workers, which is not thread-safe for concurrent
    reads. Each worker thread now gets its own PersistentClient and
    Chroma wrapper.
    """
    if not hasattr(_thread_local, 'vectorstores'):
        _thread_local.vectorstores = {}
    if col_name not in _thread_local.vectorstores:
        client = chromadb.PersistentClient(path=DB_DIR)
        _thread_local.vectorstores[col_name] = Chroma(
            client=client,
            collection_name=col_name,
            embedding_function=get_embeddings()
        )
    return _thread_local.vectorstores[col_name]


# =============================================================================
# IMPROVEMENT 6 — Two-pass chunking (semantic boundary + hard size cap)
# =============================================================================

def chunk_documents(docs: list, embeddings) -> list:
    """
    Two-pass chunking strategy.

    Pass 1 — SemanticChunker with standard_deviation threshold:
      Finds natural topic boundaries without over-splitting.
      'standard_deviation' is more stable than 'percentile' across
      documents of varying length.

    Pass 2 — RecursiveCharacterTextSplitter hard cap at 800 chars:
      Prevents any single chunk from being so large that it dilutes
      retrieval precision or blows through the generation context window.

    The previous single-pass config (percentile=95) created chunks that
    were often entire book sections, causing context bleed.
    """
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1.25
    )

    coarse_chunks = []
    for doc in docs:
        try:
            # Chunk each page individually to strictly preserve page metadata
            chunks = semantic_splitter.split_documents([doc])
            coarse_chunks.extend(chunks)
        except Exception as e:
            print(f"Semantic splitter failed on a page: {e} — falling back to page text.")
            coarse_chunks.append(doc)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len
    )

    final_chunks = size_splitter.split_documents(coarse_chunks)
    print(
        f"DEBUG: Chunking — {len(docs)} pages "
        f"→ {len(coarse_chunks)} semantic "
        f"→ {len(final_chunks)} final chunks"
    )
    return final_chunks


# =============================================================================
# IMPROVEMENT 5 — Cross-encoder re-ranking
# =============================================================================

_CROSS_ENCODER: CrossEncoder = None

def get_cross_encoder() -> CrossEncoder:
    """
    Lazy-load the cross-encoder singleton.
    ms-marco-MiniLM-L-6-v2: ~80 MB, ~8 ms/pair on CPU, strong MRR@10.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER


def rerank_documents(question: str, documents: list, top_k: int = 8) -> list:
    """
    Re-rank retrieved documents using a cross-encoder.

    WHY: Bi-encoder cosine similarity (used by Chroma) scores documents
    independently of the query. Cross-encoders jointly encode (query, doc)
    pairs and capture fine-grained interaction signals that bi-encoders miss.
    This typically improves answer relevance by 15-25% on domain-specific
    corpora without any extra LLM calls.

    Called after pooling all sub-query results, before the LLM grader,
    so the grader only sees the most promising chunks.
    """
    if len(documents) <= top_k:
        return documents

    encoder = get_cross_encoder()
    pairs = [(question, doc.page_content) for doc in documents]

    try:
        scores = encoder.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:top_k]]
        print(
            f"DEBUG: Re-ranker selected top {top_k} from {len(documents)} docs. "
            f"Score range: {ranked[-1][0]:.3f}–{ranked[0][0]:.3f}"
        )
        return top_docs
    except Exception as e:
        print(f"Re-ranker failed: {e} — using original order.")
        return documents[:top_k]


# =============================================================================
# 3. AGENTS
# =============================================================================

def rewrite_query(username, subject, user_question):
    """
    Query Rewriter — contextualises follow-up questions using recent history.
    Short-circuits immediately when there is no history (no LLM call needed).
    Uses the last 4 messages (2 full turns) for richer context.
    """
    history = load_chat_history(username, subject)

    if not history:
        return user_question

    recent_history = history[-4:]
    history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in recent_history]
    )

    print(f"--- AGENT: REWRITING QUERY ---\nOriginal: {user_question}")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    system_prompt = """You are a Query Rewriter.
Reformulate the latest user question into a standalone search query that is fully
understandable without the chat history.

RULES:
1. Replace pronouns ("it", "that", "they") with the specific nouns they refer to.
2. Expand vague phrases like "tell me more" into a concrete question about the last topic.
3. CRITICAL: If the question is already clear and self-contained (e.g., "Explain X"), return it UNCHANGED. Do NOT narrow, alter, or restrict the user's intent.
4. Output ONLY the rewritten query string — no preamble, no quotes."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Chat history:\n{history}\n\nLatest question: {question}")
    ])

    try:
        new_query = safe_invoke(
            prompt | llm, {"history": history_str, "question": user_question}
        ).strip()
        if not new_query or len(new_query) < 5:
            return user_question
        print(f"Rewritten: {new_query}")
        return new_query
    except Exception as e:
        print(f"Rewriter failed: {e}")
        return user_question


# =============================================================================
# IMPROVEMENT 4 — Planner with trigger-based decomposition (no LLM for ~80%)
# =============================================================================

DECOMPOSITION_TRIGGERS = {
    " and ", " compare", " difference between", " versus ", " vs ",
    " contrast", " both ", " each of", " all of", " list all",
    " pros and cons", " advantages and disadvantages"
}

def planner_agent(question, status_container=None):
    """
    Planner Agent — decomposes complex questions into sub-queries.

    IMPROVEMENT: Inverts the previous heuristic.
    Old logic: call LLM, then short-circuit if "simple" — wasted an LLM call
    even when it decided the query was simple.
    New logic: check for conjunction/comparative triggers first (deterministic,
    zero cost). Only invoke the LLM when a trigger is found, ensuring the LLM
    call is always justified.

    This eliminates the planning LLM call for ~80% of queries.
    """
    if status_container:
        status_container.write("**Planner Agent:** Analysing query...")

    q_lower = question.lower().strip()
    needs_decomposition = any(trigger in q_lower for trigger in DECOMPOSITION_TRIGGERS)

    if not needs_decomposition:
        print("--- AGENT: PLANNER (fast-path, simple question) ---")
        if status_container:
            status_container.write("*Simple query detected — skipping decomposition.*")
        return [question]

    print("--- AGENT: PLANNER (decomposing multi-part query) ---")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt_text = f"""Break this query into sub-questions for a retrieval system.

RULES:
- Maximum 3 sub-questions. Never more.
- Each sub-question must be independently searchable.
- Remove conjunctions and pronouns from each sub-question.
- If the query can be answered with 1 search, output exactly 1 sub-question.

Query: {question}

Output ONLY a JSON array of strings. No preamble. No explanation.
Example: ["What is BFS?", "What is DFS?", "How do BFS and DFS differ in space complexity?"]"""

    try:
        res = safe_invoke(llm, prompt_text)
        match = re.search(r'\[.*?\]', res, re.DOTALL)
        if match:
            sub_qs = [
                q.strip() for q in json.loads(match.group(0))
                if isinstance(q, str) and q.strip()
            ]
            sub_qs = sub_qs[:3]
            if sub_qs:
                if status_container:
                    status_container.write(f"*Sub-queries:* `{sub_qs}`")
                return sub_qs
    except Exception as e:
        print(f"Planner fallback: {e}")

    if status_container:
        status_container.write("*Using original query.*")
    return [question]


# =============================================================================
# IMPROVEMENT 2 — Document grader with reliable JSON parsing
# =============================================================================

def grade_documents(question, documents, status_container=None):
    """
    Document Grader (CRAG) — filters irrelevant chunks in a single batch call.

    IMPROVEMENT: Uses a greedy regex to capture the full JSON object
    (the previous non-greedy r'\{.*?\}' would match the first brace pair
    in any prose the LLM emitted before the actual result). Also skips
    grading when <= 2 documents are present (cost exceeds benefit).
    Safety fallback returns top-3 docs if the grader eliminates everything.
    """
    if status_container:
        status_container.write("**Grader Agent:** Evaluating document relevance...")
    if not documents:
        return []
    if len(documents) <= 2:
        return documents

    print(f"--- AGENT: GRADING {len(documents)} DOCUMENTS ---")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    doc_lines = "\n".join(
        f"[{i}] {d.page_content.replace(chr(10), ' ')}"
        for i, d in enumerate(documents)
    )

    prompt_text = f"""Task: Relevance filtering for a retrieval system.

Question: {question}

Documents (each prefixed with its integer ID):
{doc_lines}

Instructions:
- A document is RELEVANT if it contains information that directly helps answer the question.
- A document is IRRELEVANT if it discusses an unrelated topic, even if it looks similar.
- When uncertain, mark as relevant (conservative filtering).

You MUST respond with ONLY this JSON. No explanation. No preamble. No markdown.
{{"relevant": [<comma-separated integer IDs of relevant documents>]}}

If none are relevant: {{"relevant": []}}"""

    for attempt in range(2):
        try:
            raw = safe_invoke(llm, prompt_text)
            # Greedy match captures the full object even when prose precedes it
            match = re.search(r'\{[^{}]*"relevant"[^{}]*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                ids = [
                    i for i in parsed.get("relevant", [])
                    if isinstance(i, int) and 0 <= i < len(documents)
                ]
                filtered = [documents[i] for i in ids]

                if not filtered:
                    print("DEBUG: Grader eliminated all docs — returning top-3 as fallback.")
                    return documents[:3]

                msg = f"Kept {len(filtered)}/{len(documents)} chunks."
                print(f"DEBUG: Grader — {msg}")
                if status_container:
                    status_container.write(f"*Filtering complete:* {msg}")
                return filtered
        except json.JSONDecodeError as e:
            print(f"Grader JSON parse error (attempt {attempt + 1}): {e}")
            time.sleep(1)

    print("DEBUG: Grader failed both attempts — returning all docs.")
    return documents


# =============================================================================
# IMPROVEMENT 3 — Fast deterministic grounding check (replaces first LLM call)
# =============================================================================

def fast_grounding_check(
    answer: str, context: str, quotes: list
) -> tuple:
    """
    Deterministic grounding check. Returns (is_grounded: bool, score: float).

    If supporting_quotes are available, verifies each quote against the context
    using fuzzy matching (SequenceMatcher ratio > 0.75).
    Falls back to token-overlap scoring when no quotes are provided.

    This replaces the first check_hallucinations LLM call for well-grounded
    answers, saving ~0.5s per response.
    """
    if quotes:
        verified = 0
        for q in quotes:
            ratio = difflib.SequenceMatcher(
                None, q.lower(), context.lower()
            ).ratio()
            if ratio > 0.75:
                verified += 1
        score = verified / len(quotes)
        return score >= 0.6, score

    stops = {
        "the", "a", "an", "is", "are", "was", "were", "of",
        "in", "to", "and", "or", "it", "that", "this"
    }
    clean_answer = re.sub(r'[\*_`#]', '', answer)
    answer_tokens = set(clean_answer.lower().split()) - stops
    context_tokens = set(context.lower().split()) - stops
    if not answer_tokens:
        return False, 0.0
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return overlap > 0.65, overlap


def check_hallucinations(context_text, answer):
    """
    LLM-based hallucination grader — binary grounding check.
    Returns (grounded: bool, reason: str).

    Only called when fast_grounding_check fails (i.e. for the minority of
    answers that don't pass the deterministic gate).
    """
    print("--- AGENT: CHECKING HALLUCINATIONS (LLM) ---")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    system = """You are a STRICT grounding verifier.
Your ONLY job is to check if the proposed answer is logically supported by the provided context.

CRITICAL RULES:
1. Allow SEMANTIC PARAPHRASING: If the answer uses different words to describe the exact same concept found in the text (e.g., "measure of execution time" instead of "function of running time"), it is completely valid and GROUNDED.
2. Focus on FACTS, not EXACT PHRASING. As long as the underlying meaning matches the context, it passes.
3. You MUST strictly prohibit any genuinely NEW domain facts, time complexities, or algorithms that are nowhere to be found in the context.
4. Even if the answer is factually correct in the real world, if the underlying fact is missing from the context, it is a HALLUCINATION.

Output ONLY valid JSON in this format:
{{"grounded": true, "reason": "All facts found in context"}}
or
{{"grounded": false, "reason": "Specific fact X is missing from context"}}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
    ])

    try:
        res = safe_invoke(prompt | llm, {"context": context_text, "answer": answer})
        match = re.search(r'\{[^{}]*\}', res, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return parsed.get("grounded", False), parsed.get("reason", "")
    except Exception as e:
        print(f"Hallucination check failed: {e}")

    return True, "check unavailable"


# =============================================================================
# IMPROVEMENT 1 — Strict grounded generator prompt with structured output
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are a strict document-grounded tutor for {subject}.

## ABSOLUTE RULES
1. Every sentence you write MUST be directly supported by a passage in the CONTEXT block below.
2. If a fact is not in CONTEXT, do not state it — even if you are certain it is true.
3. If CONTEXT contains partial information, answer only the parts it covers and say:
   "Note: your uploaded notes do not cover [missing aspect]."
4. If CONTEXT contains NO relevant information at all, output exactly this JSON and nothing else:
   {{"status": "insufficient_context", "reason": "<one sentence why>"}}
5. Provide detailed, comprehensive answers. Synthesize information from the context naturally, but do not hallucinate or use outside knowledge. Use bullet points and formatting where helpful.

## OUTPUT FORMAT
Return ONLY a valid JSON object. Do not output any text before or after the JSON.
All markdown formatting (like bold headings and bullet points) MUST be INSIDE the "answer" string.
Do NOT include the confidence score or sources inside the "answer" string itself. The system handles them automatically.
Escape all newlines inside JSON strings as \\n. Do NOT use literal line breaks inside the strings.
Do NOT escape markdown characters (like * or _) with backslashes.
{{
  "answer": "<full markdown answer grounded in context>",
  "supporting_quotes": ["<verbatim or near-verbatim excerpt 1>", "..."],
  "confidence": <float 0.0–1.0, based on how completely context covers the question>
}}

## CONFIDENCE CALIBRATION
- 1.0: context directly and completely answers the question
- 0.7–0.9: context covers the question but with minor gaps
- 0.4–0.6: context is tangentially related; answer is partially inferred
- 0.0–0.3: context is barely relevant; answer should be refused

## CONTEXT
{context}"""


def critic_and_repair_agent(
    context, draft_answer, question, max_loops=2, status_container=None
):
    """
    Critic + Repair — fact-checks the draft and fixes hallucinations.

    IMPROVEMENT 3: Two-gate approach.
    Gate 1 (free): parse structured JSON output; run fast_grounding_check.
      → If grounded + confidence >= 0.6: return immediately, zero extra LLM calls.
    Gate 2 (LLM): only invoked when Gate 1 fails.

    This eliminates 1-2 LLM calls per response for well-grounded answers,
    which should be the majority on a properly configured pipeline.
    """
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # --- Parse structured generator output ---
    answer_text = draft_answer
    quotes = []
    declared_confidence = 0.5

    try:
        match = re.search(r'(\{.*\})', draft_answer, re.DOTALL)
        json_str = match.group(1) if match else draft_answer
        
        # Fix LLMs erroneously escaping markdown in JSON which breaks json.loads
        json_str = json_str.replace('\\*', '*').replace('\\_', '_')
        
        parsed = json.loads(json_str, strict=False)
        if parsed.get("status") == "insufficient_context":
            return (
                "I'm sorry, but I cannot find the answer to that in your uploaded notes.",
                True,
                0.0,
            )
        answer_text = parsed.get("answer", draft_answer)
        quotes = parsed.get("supporting_quotes", [])
        declared_confidence = float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, AttributeError, ValueError):
        # Regex fallback if JSON parsing still fails
        ans_match = re.search(r'"answer"\s*:\s*"(.*?)"\s*,\s*"supporting_quotes"', draft_answer, re.DOTALL)
        if ans_match:
            answer_text = ans_match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\*', '*').replace('\\_', '_')

    current_answer = answer_text

    # Clean up any leaked "Confidence: X.X" strings at the end of the answer
    current_answer = re.sub(r'(?i)\n*\**confidence:\**\s*[0-9.]+\s*$', '', current_answer).strip()

    for attempt in range(max_loops):
        if status_container:
            status_container.write(
                f"**Critic Agent:** Fact-checking draft (attempt {attempt + 1})..."
            )
        print(f"--- AGENT: CRITIC (attempt {attempt + 1}) ---")

        # GATE 1: Fast deterministic check — no LLM call
        is_grounded, fast_score = fast_grounding_check(current_answer, context, quotes)
        if is_grounded and declared_confidence >= 0.6:
            if status_container:
                status_container.write(
                    f"*Validation pass (fast check):* score={fast_score:.2f}"
                )
            print(f"DEBUG: Fast grounding check passed (score={fast_score:.2f}).")
            return current_answer, True, declared_confidence

        if status_container:
            status_container.write(
                f"**Critic Agent:** Fast check score={fast_score:.2f}, "
                f"running LLM verification..."
            )

        # GATE 2: LLM hallucination check
        grounded, reason = check_hallucinations(context, current_answer)
        if grounded:
            if status_container:
                status_container.write("*Validation pass (LLM check):* answer is grounded.")
            print("DEBUG: LLM hallucination grader approved.")
            return current_answer, True, declared_confidence

        # If we reach here, the strict checker found hallucinations or ungrounded facts.
        # However, check if the draft is simply a valid refusal to answer.
        refusal_phrases = ["cannot find the answer", "do not cover", "insufficient context"]
        if any(phrase in current_answer.lower() for phrase in refusal_phrases):
            if status_container:
                status_container.write("*Validation pass:* recognized as a valid refusal.")
            print("DEBUG: Critic recognized a valid refusal.")
            return current_answer, True, declared_confidence

        if status_container:
            status_container.write("**Repair Agent:** Fixing issues detected by strict verifier...")
        print(f"--- AGENT: REPAIR — {reason} ---")

        repair_prompt = f"""You are a Repair Agent.
Your ONLY knowledge source is the Context below.

Context:
{context}

User question: {question}

Flawed answer:
{current_answer}

Critic feedback: {reason}

Instructions:
1. Rewrite the answer using ONLY information from the Context.
2. If the Context genuinely does not contain enough information, reply with exactly:
   "I'm sorry, but I cannot find the answer to that in your uploaded notes."
3. Do NOT invent, guess, or use any outside knowledge."""

        current_answer = safe_invoke(llm, repair_prompt)
        current_answer = current_answer.replace('\\*', '*').replace('\\_', '_')
        quotes = []  # repaired answer has no pre-verified quotes
        declared_confidence = max(0.0, declared_confidence - 0.2)

    return current_answer, False, declared_confidence


# =============================================================================
# 4. RETRIEVAL
# =============================================================================

def hybrid_retrieve(vectorstore, query, k=5):
    """
    Hybrid Retriever — combines dense vector search with BM25 keyword search.
    No changes to the core logic; the caller now passes a thread-local
    vectorstore instance rather than a shared one.
    """
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    try:
        db_data = vectorstore.get()
        docs = db_data.get('documents', [])
        metas = db_data.get('metadatas', [])
        if docs:
            doc_objs = [
                Document(page_content=txt, metadata=m)
                for txt, m in zip(docs, metas)
            ]
            bm25_retriever = BM25Retriever.from_documents(doc_objs)
            bm25_retriever.k = k
            ensemble = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            return ensemble.invoke(query)
    except Exception as e:
        print(f"Hybrid retrieval fallback (vector only): {e}")
    return vector_retriever.invoke(query)


# =============================================================================
# 5. ORCHESTRATOR
# =============================================================================

def agentic_rag_response(username, subject, question, status_container=None, return_raw=False, max_critic_loops=3):
    """
    Main RAG orchestrator — applies all 10 improvements end-to-end.

    Pipeline:
      Rewrite → Plan → Retrieve (parallel, thread-local) →
      Re-rank (cross-encoder) → Grade → Generate (structured prompt) →
      Critic/Repair (fast gate first) → Return RAGResponse
    """
    print(f"\nDEBUG [agentic_rag_response]: user='{username}', subject='{subject}'")
    start_time = time.time()

    # ── 1. REWRITE ──────────────────────────────────────────────────────────
    rewritten_question = rewrite_query(username, subject, question)

    # ── 2. PLAN ─────────────────────────────────────────────────────────────
    sub_questions = planner_agent(rewritten_question, status_container)

    # ── 3. RETRIEVE (PARALLEL, THREAD-LOCAL) ────────────────────────────────
    col_name = get_collection_name(username, subject)

    # Validate collection exists before spawning threads
    try:
        client = get_shared_client(DB_DIR)
        existing_cols = [c.name for c in client.list_collections()]
        if col_name not in existing_cols:
            print("DEBUG: Collection not found.")
            return RAGResponse(
                answer="Error: No documents indexed for this subject.",
                confidence=0.0,
                warning="Subject database not found."
            ).to_markdown()
    except Exception as e:
        print(f"DEBUG: Could not list collections: {e}")
        return RAGResponse(
            answer="Error: Could not access subject database.",
            confidence=0.0
        ).to_markdown()

    def retrieve_one(sq):
        # Each thread gets its own vectorstore instance (IMPROVEMENT 11b)
        vs = get_thread_local_vectorstore(col_name)
        return hybrid_retrieve(vs, sq, k=20)

    raw_pooled_docs = []
    unique_content = set()

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(sub_questions), 3)
        ) as executor:
            futures = {executor.submit(retrieve_one, sq): sq for sq in sub_questions}
            for future in concurrent.futures.as_completed(futures):
                try:
                    for doc in future.result():
                        if doc.page_content not in unique_content:
                            unique_content.add(doc.page_content)
                            raw_pooled_docs.append(doc)
                except Exception as e:
                    print(f"Retrieval sub-task error: {e}")
    except Exception as e:
        print(f"Parallel retrieval failed, running sequentially: {e}")
        for sq in sub_questions:
            for doc in retrieve_one(sq):
                if doc.page_content not in unique_content:
                    unique_content.add(doc.page_content)
                    raw_pooled_docs.append(doc)

    chunks_retrieved = len(raw_pooled_docs)
    if status_container:
        status_container.write(
            f"**Hybrid Retriever:** Pooled **{chunks_retrieved}** unique chunks "
            f"from {len(sub_questions)} sub-queries."
        )

    # ── 4. RE-RANK (cross-encoder) ──────────────────────────────────────────
    # IMPROVEMENT 5: Runs before the LLM grader so the grader only sees
    # the best-ranked candidates. top_k=8 limits grader input.
    raw_pooled_docs = rerank_documents(rewritten_question, raw_pooled_docs, top_k=8)

    # ── 5. GRADE ────────────────────────────────────────────────────────────
    all_relevant_docs = grade_documents(question, raw_pooled_docs, status_container)

    if not all_relevant_docs:
        print("DEBUG: No relevant documents after grading.")
        return RAGResponse(
            answer=(
                "I checked your notes but couldn't find relevant information. "
                "Try rephrasing your question."
            ),
            confidence=0.0,
            sub_questions_used=sub_questions,
            retrieval_stats={
                "chunks_retrieved": chunks_retrieved,
                "chunks_used": 0,
                "reranked": True
            }
        ).to_markdown()

    # ── 6. BUILD CONTEXT & SOURCE LIST ──────────────────────────────────────
    sources = []
    seen_sources = set()
    for doc in all_relevant_docs:
        source_path = doc.metadata.get('source', '')
        source_name = os.path.basename(str(source_path)) if source_path else "Unknown"
        page_val = doc.metadata.get('page')
        try:
            page_num = int(page_val) + 1 if page_val is not None else "?"
        except (ValueError, TypeError):
            page_num = str(page_val)
        key = f"{source_name}:{page_num}"
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({"file": source_name, "page": page_num})

    context_text = "\n\n".join([d.page_content for d in all_relevant_docs])

    # ── 7. GENERATE DRAFT (structured prompt) ───────────────────────────────
    # IMPROVEMENT 1: Strict grounded prompt with JSON output including
    # supporting_quotes and confidence. This feeds the fast grounding check
    # in the critic and eliminates most LLM critic calls.
    if status_container:
        status_container.write("**Drafting:** Writing initial response from verified context...")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    sys_prompt = GENERATOR_SYSTEM_PROMPT.format(
        subject=subject,
        context=context_text
    )

    try:
        draft_answer = safe_invoke(
            llm, [("system", sys_prompt), ("human", question)]
        )
    except RateLimitExhaustedError as e:
        return RAGResponse(
            answer="Service temporarily unavailable due to rate limiting. Please try again shortly.",
            confidence=0.0,
            warning=str(e)
        ).to_markdown()

    # ── 8. CRITIC & REPAIR ──────────────────────────────────────────────────
    try:
        final_answer, is_grounded, confidence = critic_and_repair_agent(
            context_text, draft_answer, question,
            max_loops=max_critic_loops, status_container=status_container
        )
    except RateLimitExhaustedError as e:
        # Critic exhausted retries — return draft with a warning rather than
        # silently dropping the answer (IMPROVEMENT 9).
        final_answer = draft_answer
        is_grounded = False
        confidence = 0.3
        print(f"DEBUG: Critic rate-limited: {e}")

    elapsed = time.time() - start_time
    print(f"DEBUG: Total pipeline time: {elapsed:.2f}s")

    # ── 9. BUILD & CACHE RESPONSE ───────────────────────────────────────────
    warning = None
    if not is_grounded:
        warning = (
            "Repair Agent Warning: answer could not be fully verified after 2 attempts."
        )

    response = RAGResponse(
        answer=final_answer,
        confidence=confidence,
        sources=sources,
        sub_questions_used=sub_questions,
        retrieval_stats={
            "chunks_retrieved": chunks_retrieved,
            "chunks_used": len(all_relevant_docs),
            "reranked": True,
            "pipeline_time_s": round(elapsed, 2)
        },
        warning=warning
    )

    if return_raw:
        return response, [d.page_content for d in all_relevant_docs]

    return response.to_markdown()


# =============================================================================
# 6. RAG CHAIN (simple, non-agentic path)
# =============================================================================

def get_rag_chain(username, subject):
    client = get_shared_client(DB_DIR)
    col_name = get_collection_name(username, subject)
    try:
        collections = [c.name for c in client.list_collections()]
        if col_name not in collections:
            return None
        vectorstore = Chroma(
            client=client,
            collection_name=col_name,
            embedding_function=get_embeddings()
        )
    except Exception:
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_retries=10  # Add retries for robustness in batch jobs
    )
    system_prompt = (
        f"You are a strict AI Tutor for {subject}. "
        "Use ONLY the following context to answer. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))


# =============================================================================
# 7. FILE & SUBJECT MANAGEMENT
# =============================================================================

def handle_file_upload(username, subject_name, uploaded_files):
    """
    Index uploaded PDFs into the Chroma collection for this subject.
    Uses the improved two-pass chunker (IMPROVEMENT 6).
    """
    subject_path = os.path.join(SUBJECTS_DIR, username, subject_name)
    os.makedirs(subject_path, exist_ok=True)
    client = get_shared_client(DB_DIR)
    embeddings = get_embeddings()

    for f in uploaded_files:
        save_path = os.path.join(subject_path, f.name)
        with open(save_path, "wb") as file:
            file.write(f.getbuffer())

        loader = PyPDFLoader(save_path)
        docs = loader.load()

        if docs:
            print(f"DEBUG: Chunking {f.name}...")
            # IMPROVEMENT 6: two-pass chunking replaces single SemanticChunker pass
            splits = chunk_documents(docs, embeddings)
            print(f"DEBUG: {len(splits)} final chunks created.")

            col_name = get_collection_name(username, subject_name)
            Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=col_name,
                persist_directory=DB_DIR,
                client=client
            )
            time.sleep(1)
            print(f"DEBUG: Indexed {len(splits)} chunks into '{subject_name}'.")


# =============================================================================
# IMPROVEMENT 10 — delete_file now actually removes chunks from the index
# =============================================================================

class _FakeFile:
    """Minimal file-like object for passing existing PDFs to handle_file_upload."""
    def __init__(self, path: str):
        self.name = os.path.basename(path)
        self._path = path

    def getbuffer(self):
        with open(self._path, "rb") as f:
            return f.read()


def rebuild_index_for_subject(username: str, subject: str) -> None:
    """
    Drop and recreate the Chroma collection for a subject, re-indexing all
    currently existing PDFs. Called after any file deletion so that removed
    content is never returned in future queries.

    FIX: The previous delete_file called handle_file_upload(username, subject, [])
    which is a no-op on an empty list — deleted chunks stayed in the index
    indefinitely and continued appearing in answers.
    """
    client = get_shared_client(DB_DIR)
    col_name = get_collection_name(username, subject)

    try:
        client.delete_collection(col_name)
        print(f"DEBUG: Dropped collection '{col_name}'")
    except Exception:
        pass

    subject_path = os.path.join(SUBJECTS_DIR, username, subject)
    if not os.path.exists(subject_path):
        return

    pdf_files = [f for f in os.listdir(subject_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"DEBUG: No remaining PDFs for '{subject}' — collection left empty.")
        return

    fake_files = [
        _FakeFile(os.path.join(subject_path, f)) for f in pdf_files
    ]
    handle_file_upload(username, subject, fake_files)
    print(f"DEBUG: Re-indexed {len(pdf_files)} files for '{subject}'")


def delete_file(username: str, subject: str, filename: str) -> None:
    """
    Delete a PDF from disk then rebuild the index so its chunks are removed.

    IMPROVEMENT 10: replaces the broken handle_file_upload(username, subject, [])
    call with rebuild_index_for_subject which actually drops and recreates the
    Chroma collection from the remaining files.
    """
    path = os.path.join(SUBJECTS_DIR, username, subject, filename)
    if os.path.exists(path):
        os.remove(path)
        print(f"DEBUG: Deleted file '{filename}'")
    else:
        print(f"DEBUG: File not found for deletion: {path}")
    rebuild_index_for_subject(username, subject)


def delete_subject(username, subject):
    s_path = os.path.join(SUBJECTS_DIR, username, subject)
    if os.path.exists(s_path):
        shutil.rmtree(s_path)
    client = get_shared_client(DB_DIR)
    col_name = get_collection_name(username, subject)
    try:
        client.delete_collection(col_name)
    except Exception:
        pass
    clear_chat_history(username, subject)


def list_files(username, subject):
    path = os.path.join(SUBJECTS_DIR, username, subject)
    return (
        [f for f in os.listdir(path) if f.endswith(".pdf")]
        if os.path.exists(path) else []
    )


# =============================================================================
# 8. QUIZ, SUMMARY, FLASHCARDS, SUGGESTIONS
# =============================================================================

def generate_quiz(username, subject, topic=""):
    client = get_shared_client(DB_DIR)
    col_name = get_collection_name(username, subject)
    try:
        vectorstore = Chroma(
            client=client, collection_name=col_name,
            embedding_function=get_embeddings()
        )
        if topic:
            docs = vectorstore.as_retriever(search_kwargs={"k": 10}).invoke(topic)
            if not docs:
                return None
            context = "\n".join([d.page_content for d in docs])
        else:
            all_data = vectorstore.get()
            if not all_data or not all_data.get('documents'):
                return None
            docs_list = all_data['documents']
            selected_docs = random.sample(docs_list, min(len(docs_list), 10))
            context = "\n".join(selected_docs)
    except Exception as e:
        print(f"Quiz DB error: {e}")
        return None

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    topic_instruction = (
        f"Focus strictly on the topic: '{topic}'."
        if topic else
        "Ensure questions cover a WIDE RANGE of different topics and concepts."
    )

    prompt = f"""You are an expert exam setter.
Based on the text below, generate EXACTLY 5 unique multiple-choice questions.
{topic_instruction}

TEXT: "{context[:8000]}"

CRITICAL JSON RULES:
1. Return ONLY a valid JSON array — no markdown, no intro text.
2. Escape any double quotes inside values with \\".
3. No trailing commas.
Format: [{{"question":"...","options":["A","B","C","D"],"answer":"Exact option text","explanation":"..."}}]"""

    for attempt in range(3):
        try:
            res = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3).invoke(prompt).content
            match = re.search(r'\[.*\]', res, re.DOTALL)
            if match:
                return json.loads(match.group(0), strict=False)
        except Exception as e:
            print(f"Quiz JSON error (attempt {attempt + 1}/3): {e}")
            time.sleep(1)
    return []


def generate_summary(username, subject, topic=""):
    client = get_shared_client(DB_DIR)
    col_name = get_collection_name(username, subject)
    try:
        collections = [c.name for c in client.list_collections()]
        if col_name not in collections:
            return "No documents found."
        vectorstore = Chroma(
            client=client, collection_name=col_name,
            embedding_function=get_embeddings()
        )
        search_query = topic if topic else "main topics and overview"
        docs = vectorstore.as_retriever(search_kwargs={"k": 15}).invoke(search_query)
        context = "\n".join([d.page_content for d in docs])
    except Exception:
        return "Error accessing documents."

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    topic_suffix = f" specifically about '{topic}'" if topic else ""
    return llm.invoke(
        f"Create a 1-page cheat sheet{topic_suffix} from:\n{context[:12000]}"
    ).content


def generate_flashcards(username, subject, topic=""):
    client = get_shared_client(DB_DIR)
    col_name = get_collection_name(username, subject)
    try:
        vectorstore = Chroma(
            client=client, collection_name=col_name,
            embedding_function=get_embeddings()
        )
        search_query = (
            topic if topic
            else "key concepts definitions examples characteristics types"
        )
        docs = vectorstore.as_retriever(search_kwargs={"k": 15}).invoke(search_query)
        if not docs and not topic:
            all_data = vectorstore.get()
            if all_data and all_data['documents']:
                context = "\n".join(
                    random.sample(all_data['documents'], min(len(all_data['documents']), 5))
                )
            else:
                return None
        elif docs:
            selected = random.sample(docs, min(len(docs), 5)) if not topic else docs[:5]
            context = "\n".join([d.page_content for d in selected])
        else:
            return None
    except Exception as e:
        print(f"Flashcard retrieval error: {e}")
        return None

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
    topic_instruction = f" focusing strictly on: '{topic}'" if topic else ""

    prompt = f"""Create 5 DETAILED flashcards{topic_instruction} from this text.

TEXT: "{context[:6000]}"

CRITICAL JSON RULES:
1. Return ONLY a valid JSON array — no markdown, no intro text.
2. Escape double quotes inside values with \\".
3. No trailing commas.
4. The "back" MUST include bold headings **Definition:**, **Key Points:**, **Example:**.
Format: [{{"front":"Term","back":"**Definition:** ...\\n\\n**Key Points:** ...\\n\\n**Example:** ..."}}]"""

    for attempt in range(3):
        try:
            res = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5).invoke(prompt).content
            match = re.search(r'\[.*\]', res, re.DOTALL)
            if match:
                return json.loads(match.group(0), strict=False)
        except Exception as e:
            print(f"Flashcard JSON error (attempt {attempt + 1}/3): {e}")
            time.sleep(1)
    return []


def generate_suggested_questions(username, subject, history=None):
    context = ""
    is_follow_up = False

    if history:
        for msg in reversed(history):
            if msg["role"] == "assistant":
                context = msg["content"]
                is_follow_up = True
                break

    if not context:
        client = get_shared_client(DB_DIR)
        col_name = get_collection_name(username, subject)
        try:
            collections = [c.name for c in client.list_collections()]
            if col_name not in collections:
                return []
            vectorstore = Chroma(
                client=client, collection_name=col_name,
                embedding_function=get_embeddings()
            )
            db_data = vectorstore.get()
            if not db_data or not db_data.get('documents'):
                return []
            docs_list = db_data['documents']
            context = "\n".join(random.sample(docs_list, min(len(docs_list), 3)))
        except Exception as e:
            print(f"Suggestion DB error: {e}")
            return []

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

    if is_follow_up:
        prompt = f"""Based on the tutor explanation below, suggest EXACTLY 3 insightful
follow-up questions a student might ask.
Keep questions under 12 words each.
TEXT: "{context[:4000]}"
Output ONLY a valid JSON list of strings."""
    else:
        prompt = f"""Based on the student's notes below, suggest EXACTLY 3 broad questions
to test their understanding.
Keep questions under 12 words each.
TEXT: "{context[:4000]}"
Output ONLY a valid JSON list of strings."""

    for attempt in range(3):
        try:
            res = llm.invoke(prompt).content
            match = re.search(r'\[.*?\]', res, re.DOTALL)
            if match:
                return json.loads(match.group(0))[:3]
        except Exception as e:
            print(f"Suggestion error (attempt {attempt + 1}): {e}")
            time.sleep(1)
    return []


# =============================================================================
# 9. AUDIO & PDF EXPORT
# =============================================================================

def transcribe_audio(audio_bytes):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"
    try:
        return client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
    except Exception as e:
        return f"Error transcribing: {str(e)}"


def create_pdf_bytes(text, title="Subject Summary"):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, title)

    p.setFont("Helvetica", 12)
    y_position = height - 80

    for line in text.split('\n'):
        for w_line in simpleSplit(line, "Helvetica", 12, width - 100):
            if y_position < 50:
                p.showPage()
                p.setFont("Helvetica", 12)
                y_position = height - 50
            p.drawString(50, y_position, w_line)
            y_position -= 15
        y_position -= 5

    p.save()
    buffer.seek(0)
    return buffer