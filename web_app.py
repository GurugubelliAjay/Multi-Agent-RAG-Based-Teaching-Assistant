import os
from dotenv import load_dotenv
load_dotenv()

# --- STANDARD IMPORTS ---
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from langchain_groq import ChatGroq
import requests
from streamlit_lottie import st_lottie
import streamlit_shadcn_ui as ui
import streamlit_antd_components as sac

@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

# ... rest of your imports
from agent_tools import (get_rag_chain, handle_file_upload, delete_file, list_files, 
                         delete_subject, generate_quiz, generate_summary, 
                         load_chat_history, save_message, clear_chat_history, 
                         generate_flashcards, transcribe_audio,
                         register_user, authenticate_user,
                         create_session, get_username_from_session, destroy_session)

st.set_page_config(page_title="Student Hub", page_icon=":material/school:", layout="wide")

# --- MULTI-USER AUTHENTICATION ---
if "authenticated" not in st.session_state: 
    st.session_state.authenticated = False
    st.session_state.username = None

# Persistent login check via URL query parameters
if not st.session_state.authenticated and "session_token" in st.query_params:
    token = st.query_params["session_token"]
    username = get_username_from_session(token)
    if username:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.session_token = token
    else:
        del st.query_params["session_token"] # Clean up invalid/expired tokens

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        lottie_hello = load_lottieurl("https://lottie.host/65e94b43-6901-4ec9-8666-6b2256422b40/gK7P59hWw4.json")
        if lottie_hello:
            st_lottie(lottie_hello, height=200, key="hello")
        st.title("Welcome to Student Hub")
        
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
        
        with tab_login:
            with st.form("login_form", border=True):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Login", type="primary", use_container_width=True):
                    if authenticate_user(user, pwd):
                        st.session_state.authenticated = True
                        st.session_state.username = user
                        token = create_session(user)
                        st.session_state.session_token = token
                        st.query_params["session_token"] = token # Add token to URL
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                        
        with tab_signup:
            with st.form("signup_form", border=True):
                new_user = st.text_input("Choose a Username")
                new_pwd = st.text_input("Choose a Password", type="password")
                confirm_pwd = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Sign Up", type="primary", use_container_width=True):
                    if new_pwd != confirm_pwd:
                        st.error("Passwords do not match.")
                    elif len(new_user) < 3 or len(new_pwd) < 6:
                        st.error("Username must be at least 3 characters and password at least 6 characters.")
                    else:
                        if register_user(new_user, new_pwd):
                            st.success("Account created successfully! You can now log in.")
                        else:
                            st.error("Username already exists. Please choose another one.")
    st.stop()

# --- MASTER CSS STYLING ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Apply Google Font smoothly without aggressive wildcard classes */
    html, body, .stApp, h1, h2, h3, h4, h5, h6, p, label, button, input, li {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Safely protect all variations of Material Icons */
    .material-symbols-rounded, 
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
    }

    /* 1. GENERAL BUTTONS */
    .stButton button { border-radius: 8px; transition: all 0.2s; }

    /* 2. SECONDARY BUTTONS (White to match Mic Recorder) */
    button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #0f172a !important;
        font-weight: 500;
    }
    button[kind="secondary"]:hover, button[kind="secondary"]:focus {
        background-color: #f8fafc !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        outline: none !important;
    }
    
    /* 3. FIX MIC RECORDER CLIPPING (Reveals bottom border) */
    iframe[title*="mic_recorder"] {
        min-height: 45px !important;
    }

    /* 4. PRIMARY BUTTONS (Green) */
    button[kind="primary"] {
        background-color: #2da44e !important;
        border-color: #2da44e !important;
        color: white !important;
        font-weight: 600;
    }
    button[kind="primary"]:hover, button[kind="primary"]:focus {
        background-color: #2a9147 !important;
        border-color: #2a9147 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        outline: none !important;
    }
    
</style>
""", unsafe_allow_html=True)

# State Init
if "current_view" not in st.session_state: st.session_state.current_view = "Dashboard"
if "active_subject" not in st.session_state: st.session_state.active_subject = None
if "sidebar_selection" not in st.session_state: st.session_state.sidebar_selection = "Dashboard"
if "quiz_data" not in st.session_state: st.session_state.quiz_data = {}
if "summaries" not in st.session_state: st.session_state.summaries = {}
if "flashcards" not in st.session_state: st.session_state.flashcards = {}
if "flipped" not in st.session_state: st.session_state.flipped = {}
if "last_audio" not in st.session_state: st.session_state.last_audio = None

# --- SIDEBAR ---
with st.sidebar:
    # Replaced URL and increased width for the new logo
    st.image("https://d2lk14jtvqry1q.cloudfront.net/media/large_299_abb805a780_463efb8229.png", width=300)
    st.markdown("<br>", unsafe_allow_html=True)
    
    selected_nav = sac.menu([
        sac.MenuItem('Dashboard', icon='house'),
        sac.MenuItem('General Chat', icon='chat-dots'),
    ], size='md', variant='filled', color='#2da44e')
    
    if selected_nav != st.session_state.sidebar_selection:
        st.session_state.sidebar_selection = selected_nav
        if selected_nav == "Dashboard":
            st.session_state.current_view = "Dashboard"
            st.session_state.active_subject = None
        elif selected_nav == "General Chat":
            st.session_state.current_view = "Course"
            st.session_state.active_subject = "General Chat"
        st.rerun()

    st.divider()
    with st.expander("New Notebook", icon=":material/add_circle:"):
        new_sub = st.text_input("Name")
        if st.button("Create", use_container_width=True, type="primary", icon=":material/add:") and new_sub:
            os.makedirs(f"subjects/{new_sub}", exist_ok=True)
            os.makedirs(f"subjects/{st.session_state.username}/{new_sub}", exist_ok=True)
            st.rerun()
            
    st.divider()
    st.caption(f"Logged in as: **{st.session_state.username}**")
    if st.button("Logout", icon=":material/logout:", use_container_width=True):
        if "session_token" in st.session_state:
            destroy_session(st.session_state.session_token)
        if "session_token" in st.query_params:
            del st.query_params["session_token"]
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# --- DASHBOARD ---
if st.session_state.current_view == "Dashboard":
    st.title("My Notebooks")
    if not os.path.exists("subjects"): os.makedirs("subjects")
    subjects = [d for d in os.listdir("subjects") if os.path.isdir(os.path.join("subjects", d))]
    
    user_subjects_dir = os.path.join("subjects", st.session_state.username)
    if not os.path.exists(user_subjects_dir): os.makedirs(user_subjects_dir)
    subjects = [d for d in os.listdir(user_subjects_dir) if os.path.isdir(os.path.join(user_subjects_dir, d))]
    
    if not subjects: st.info("Create a notebook in the sidebar to begin!")
    
    cols = st.columns(3)
    for i, sub in enumerate(subjects):
        with cols[i % 3]:
            with st.container(border=True):
                files = list_files(st.session_state.username, sub)
                st.subheader(sub)
                st.caption(f"**{len(files)}** document{'s' if len(files) != 1 else ''} attached")
                
                if st.button("Open Notebook", key=f"btn_{sub}", use_container_width=True, type="primary"):
                    st.session_state.current_view = "Course"
                    st.session_state.active_subject = sub
                    st.rerun()

# --- COURSE VIEW ---
elif st.session_state.current_view == "Course":
    subject = st.session_state.active_subject
    
    c1, c2 = st.columns([0.8, 0.2])
    with c1: st.title(f"{subject}")
    with c2: 
        if st.button("Dashboard", use_container_width=True, type="primary", icon=":material/arrow_back:"):
            st.session_state.current_view = "Dashboard"
            st.rerun()

    col_main, col_tools = st.columns([0.7, 0.3])
    
    # Right Sidebar
    # Right Sidebar
    # Inside col_tools
    with col_tools:
        with st.container(border=True):
            st.subheader(":material/folder_open: Files")
            
            if subject != "General Chat":
                # Create a dynamic key for the uploader
                if "uploader_key" not in st.session_state:
                    st.session_state.uploader_key = 0
                
                up = st.file_uploader(
                    "Upload PDF", 
                    type="pdf", 
                    accept_multiple_files=True, 
                    key=f"uploader_{st.session_state.uploader_key}"
                )
                
                if up:
                    existing_files = list_files(st.session_state.username, subject)
                    new_files = [f for f in up if f.name not in existing_files]
                    
                    if new_files:
                        with st.spinner("Processing New Files..."):
                            handle_file_upload(st.session_state.username, subject, new_files)
                        
                        # THE FIX: Increment the key to clear the uploader UI
                        st.session_state.uploader_key += 1
                        st.rerun() 

                st.divider()
                # ... (rest of your file listing code)
                
                # Display Files Section
                files = list_files(st.session_state.username, subject)
                if not files:
                    st.caption("No files uploaded yet.")
                else:
                    for f in files:
                        fc1, fc2 = st.columns([0.8, 0.2])
                        fc1.caption(f)
                        # Red-styled delete button for files
                        fc2.markdown('<div class="danger-zone"></div>', unsafe_allow_html=True)
                        if fc2.button("", key=f"del_{f}", icon=":material/delete:", type="primary"): 
                            delete_file(st.session_state.username, subject, f)
                            st.rerun()
                
                st.divider()
                st.markdown('<div class="danger-zone"></div>', unsafe_allow_html=True)
                if st.button("Delete Notebook", use_container_width=True, icon=":material/delete_forever:"):
                    delete_subject(st.session_state.username, subject)
                    st.session_state.current_view = "Dashboard"
                    st.rerun()

    # Main Area
    with col_main:
        tabs = sac.tabs([
            sac.TabsItem(label='Chat', icon='chat-text'),
            sac.TabsItem(label='Quiz', icon='controller'),
            sac.TabsItem(label='Summaries', icon='file-earmark-text'),
            sac.TabsItem(label='Flashcards', icon='layers'),
        ], align='center', size='md', variant='outline', color='#2da44e')
        
        # TAB 1: CHAT
        if tabs == "Chat":
            history = load_chat_history(st.session_state.username, subject)
            chat_con = st.container(height=450, border=True)
            with chat_con:
                for m in history:
                    with st.chat_message(m["role"]): st.markdown(m["content"])
            
            prompt = st.chat_input("Type a message...")

            c_voice, c_reset = st.columns([0.15, 0.85])
            with c_voice:
                audio = mic_recorder(start_prompt="Speak", stop_prompt="Stop", key='recorder')
            with c_reset:
                if st.button("Clear Chat", use_container_width=True, type="secondary", icon=":material/delete_sweep:"):
                    clear_chat_history(st.session_state.username, subject)
                    st.session_state.last_audio = None 
                    st.toast(f"Chat history for {subject} cleared!")
                    st.rerun()

            final_input = None
            if prompt: final_input = prompt
            elif audio:
                if audio['bytes'] != st.session_state.last_audio:
                    st.session_state.last_audio = audio['bytes']
                    with st.spinner("Transcribing..."):
                        text = transcribe_audio(audio['bytes'])
                        if text: final_input = text
            
            if final_input:
                save_message(st.session_state.username, subject, "user", final_input)
                with chat_con:
                    with st.chat_message("user"): st.markdown(final_input)
                
                if subject == "General Chat":
                    with chat_con:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                ans = ChatGroq(model="llama-3.1-8b-instant").invoke(final_input).content
                                st.markdown(ans)
                else:
                    with chat_con:
                        with st.chat_message("assistant"):
                            with st.status("Initiating Multi-Agent Workflow...", expanded=True) as status:
                                from agent_tools import agentic_rag_response
                                ans = agentic_rag_response(st.session_state.username, subject, final_input, status_container=status)
                                
                                if "I checked your notes, but" in ans:
                                    status.update(label="No relevant notes found", state="error", expanded=False)
                                elif "Warning:" in ans: 
                                    status.update(label="Fact-check warning", state="error", expanded=False)
                                else:
                                    status.update(label="Verified Answer Generated", state="complete", expanded=False)
                            st.markdown(ans)
                
                save_message(st.session_state.username, subject, "assistant", ans)
                st.rerun()

        # TAB 2: QUIZ
        elif tabs == "Quiz":
            if subject == "General Chat": st.warning("Open a notebook.")
            else:
                if st.button("New Quiz", use_container_width=True, type="primary", icon=":material/casino:"):
                    with st.container(border=True):
                        lottie_quiz = load_lottieurl("https://lottie.host/26251b5c-4e4f-4d92-80ea-3e75b9ea8925/tC6AylD6s0.json")
                        if lottie_quiz: st_lottie(lottie_quiz, height=150)
                        st.info("Generating your personalized quiz...")
                        st.session_state.quiz_data[subject] = generate_quiz(st.session_state.username, subject)
                        st.rerun()
                
                if subject in st.session_state.quiz_data:
                    quiz = st.session_state.quiz_data[subject]
                    if not quiz: st.error("Not enough text found in PDFs to generate questions.")
                    else:
                        with st.form("quiz_form", border=True):
                            st.subheader("Knowledge Check")
                            user_answers = []
                            for i, q in enumerate(quiz):
                                st.markdown(f"**Q{i+1}: {q['question']}**")
                                choice = st.radio("Select an answer:", q['options'], key=f"quiz_opt_{i}")
                                user_answers.append(choice)
                                st.divider()
                            
                            submitted = st.form_submit_button("Submit Answers", type="primary", icon=":material/check_circle:")
                            
                        if submitted:
                            score = 0
                            with st.container(border=True):
                                st.subheader("Quiz Results")
                                for i, q in enumerate(quiz):
                                    is_correct = user_answers[i] == q['answer']
                                    if is_correct:
                                        score += 1
                                        st.success(f"**Question {i+1}: Correct!**")
                                    else:
                                        st.error(f"**Question {i+1}: Incorrect.**")
                                        st.info(f"**Correct Answer:** {q['answer']}")
                                    
                                    st.write(f"*Explanation:* {q['explanation']}")
                                    st.divider()
                                
                                ui.metric_card(title="Final Score", content=f"{score} / {len(quiz)}", description="Knowledge Check Complete", key="quiz_score")

        # TAB 3: SUMMARIES
        elif tabs == "Summaries":
            if subject == "General Chat": 
                st.warning("Open a notebook.")
            else:
                if st.button("Generate Summary", use_container_width=True, type="primary", icon=":material/bolt:"):
                    with st.spinner("Reading..."):
                        st.session_state.summaries[subject] = generate_summary(st.session_state.username, subject)
                        st.rerun()
                
                if subject in st.session_state.summaries:
                    summary_text = st.session_state.summaries[subject]
                    with st.container(border=True):
                        st.markdown(summary_text)
                    
                    st.divider()
                    
                    from agent_tools import create_pdf_bytes
                    
                    pdf_data = create_pdf_bytes(summary_text, title=f"Summary: {subject}")
                    
                    st.download_button(
                        label="Download as PDF",
                        data=pdf_data,
                        file_name=f"{subject}_Summary.pdf",
                        mime="application/pdf",
                        icon=":material/download:",
                        type="primary",
                        use_container_width=True
                    )

        # TAB 4: FLASHCARDS
        # TAB 4: FLASHCARDS
        elif tabs == "Flashcards":
            st.subheader(":material/style: Flashcards")
            if subject == "General Chat": 
                st.warning("Open a notebook to use flashcards.")
            else:
                files = list_files(st.session_state.username, subject)
                if not files:
                    st.info("Please upload some PDFs first to generate flashcards.")
                else:
                    if st.button("Generate Deck", type="primary", use_container_width=True, icon=":material/auto_awesome:"):
                        with st.container(border=True):
                            lottie_cards = load_lottieurl("https://lottie.host/bd7dc339-bb74-4b53-83ff-a1851e0bc67a/7xH4oT0x4A.json")
                            if lottie_cards: st_lottie(lottie_cards, height=150)
                            st.info("Extracting key concepts for flashcards...")
                            st.session_state.flashcards[subject] = [] 
                            
                            cards = generate_flashcards(st.session_state.username, subject)
                            if cards:
                                st.session_state.flashcards[subject] = cards
                                st.session_state.flipped = {i: False for i in range(len(cards))}
                                st.rerun()
                            else:
                                st.error("Could not find enough content. Try uploading another PDF.")
                    
                    deck = st.session_state.flashcards.get(subject, [])
                    if deck:
                        st.divider()
                        cols = st.columns(2)
                        for i, card in enumerate(deck):
                            with cols[i % 2]:
                                is_flipped = st.session_state.flipped.get(i, False)
                                with st.container(border=True):
                                    if not is_flipped:
                                        st.markdown(f"<div style='text-align: center; min-height: 150px; display: flex; align-items: center; justify-content: center;'><h3 style='color: #2da44e;'>{card['front']}</h3></div>", unsafe_allow_html=True)
                                        if st.button("Show Definition", key=f"flip_{i}_{subject}", use_container_width=True):
                                            st.session_state.flipped[i] = True
                                            st.rerun()
                                    else:
                                        st.markdown(f"<div style='min-height: 150px; padding: 10px;'>{card['back']}</div>", unsafe_allow_html=True)
                                        if st.button("Flip Back", key=f"flip_{i}_{subject}", use_container_width=True, type="primary"):
                                            st.session_state.flipped[i] = False
                                            st.rerun()
                    else:
                        st.info("No flashcards yet. Click 'Generate Deck'.")