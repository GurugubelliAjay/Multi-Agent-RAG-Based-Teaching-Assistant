import os
from dotenv import load_dotenv
load_dotenv()

# --- STANDARD IMPORTS ---
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_mic_recorder import mic_recorder
from langchain_groq import ChatGroq

# ... rest of your imports
from agent_tools import (get_rag_chain, handle_file_upload, delete_file, list_files, 
                         delete_subject, generate_quiz, generate_summary, 
                         load_chat_history, save_message, clear_chat_history, 
                         generate_flashcards, transcribe_audio)

st.set_page_config(page_title="AI Learning Hub", page_icon=":material/school:", layout="wide")

# --- MASTER CSS STYLING ---
st.markdown("""
<style>
    /* 1. GAPLESS DASHBOARD CARDS */
    div[data-testid="column"] div[data-testid="stImage"] { margin-bottom: -18px !important; }
    div[data-testid="column"] .stButton button {
        border-radius: 0px 0px 12px 12px !important;
        border-top: none !important;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
        color: #333;
        font-weight: 600;
        width: 100%;
    }
    div[data-testid="column"] .stButton button:hover { background-color: #ececec; color: #000; }
    div[data-testid="column"] img { border-radius: 12px 12px 0px 0px !important; object-fit: cover; }

    /* 2. GENERAL BUTTONS */
    .stButton button { border-radius: 8px; transition: all 0.2s; }

    /* 3. PRIMARY BUTTONS (Green) */
    button[kind="primary"] {
        background-color: #2da44e !important;
        border-color: #2da44e !important;
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #2a9147 !important;
        border-color: #2a9147 !important;
    }
    
    /* 4. FLASHCARD STYLING (Colorful & Big) */
    /* Target the buttons inside the Flashcard Tab specifically */
    /* We assume flashcards appear as a vertical stack of buttons */
    
    /* Card 1: Pink */
    div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton:nth-child(2) button {
        background-color: #FFADAD !important; color: #333 !important; border: 2px solid #FFADAD; height: 150px; font-size: 18px;
    }
    /* Card 2: Orange */
    div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton:nth-child(3) button {
        background-color: #FFD6A5 !important; color: #333 !important; border: 2px solid #FFD6A5; height: 150px; font-size: 18px;
    }
    /* Card 3: Yellow */
    div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton:nth-child(4) button {
        background-color: #FDFFB6 !important; color: #333 !important; border: 2px solid #FDFFB6; height: 150px; font-size: 18px;
    }
    /* Card 4: Green */
    div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton:nth-child(5) button {
        background-color: #CAFFBF !important; color: #333 !important; border: 2px solid #CAFFBF; height: 150px; font-size: 18px;
    }
    /* Card 5: Blue */
    div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton:nth-child(6) button {
        background-color: #9BF6FF !important; color: #333 !important; border: 2px solid #9BF6FF; height: 150px; font-size: 18px;
    }
    
    /* Hover effects for Flashcards: slight lift + shadow */
    div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
</style>
""", unsafe_allow_html=True)

# --- CONFIG ---
CARD_COLORS = [
    "FFADAD", "FFD6A5", "FDFFB6", "CAFFBF", 
    "9BF6FF", "A0C4FF", "BDB2FF", "FFC6FF",
    "FFFFFC", "B5EAD7"
]

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
    
    selected_nav = option_menu(
        menu_title=None,
        options=["Dashboard", "General Chat"],
        icons=["house", "chat-dots"],
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#2da44e"}}
    )
    
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
    with st.expander("New Course", icon=":material/add_circle:"):
        new_sub = st.text_input("Name")
        if st.button("Create", width="stretch", type="primary", icon=":material/add:") and new_sub:
            os.makedirs(f"subjects/{new_sub}", exist_ok=True)
            st.rerun()

# --- DASHBOARD ---
if st.session_state.current_view == "Dashboard":
    st.title("My Learning Hub")
    if not os.path.exists("subjects"): os.makedirs("subjects")
    subjects = [d for d in os.listdir("subjects") if os.path.isdir(os.path.join("subjects", d))]
    
    if not subjects: st.info("Create a course in the sidebar to begin!")
    
    cols = st.columns(3)
    for i, sub in enumerate(subjects):
        with cols[i % 3]:
            color = CARD_COLORS[i % len(CARD_COLORS)]
            st.image(f"https://placehold.co/600x300/{color}/{color}.png", width="stretch")
            if st.button(f"{sub}", key=f"btn_{sub}", width="stretch"):
                st.session_state.current_view = "Course"
                st.session_state.active_subject = sub
                st.rerun()

# --- COURSE VIEW ---
elif st.session_state.current_view == "Course":
    subject = st.session_state.active_subject
    
    c1, c2 = st.columns([0.85, 0.15])
    with c1: st.title(f"{subject}")
    with c2: 
        if st.button("Dashboard", width="stretch", type="primary", icon=":material/arrow_back:"):
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
                    existing_files = list_files(subject)
                    new_files = [f for f in up if f.name not in existing_files]
                    
                    if new_files:
                        with st.spinner("Processing New Files..."):
                            handle_file_upload(subject, new_files)
                        
                        # THE FIX: Increment the key to clear the uploader UI
                        st.session_state.uploader_key += 1
                        st.rerun() 

                st.divider()
                # ... (rest of your file listing code)
                
                # Display Files Section
                files = list_files(subject)
                if not files:
                    st.caption("No files uploaded yet.")
                else:
                    for f in files:
                        fc1, fc2 = st.columns([0.8, 0.2])
                        fc1.caption(f)
                        # Red-styled delete button for files
                        if fc2.button("", key=f"del_{f}", icon=":material/delete:", type="primary"): 
                            delete_file(subject, f)
                            st.rerun()
                
                st.divider()
                if st.button("Delete Course", width="stretch", icon=":material/delete_forever:"):
                    delete_subject(subject)
                    st.session_state.current_view = "Dashboard"
                    st.rerun()

    # Main Area
    with col_main:
        tabs = option_menu(None, ["Chat", "Quiz", "Summaries", "Flashcards"], 
                           icons=["chat", "pencil", "file-text", "card-checklist"], 
                           orientation="horizontal")
        
        # TAB 1: CHAT
        if tabs == "Chat":
            history = load_chat_history(subject)
            chat_con = st.container(height=450)
            with chat_con:
                for m in history:
                    with st.chat_message(m["role"]): st.markdown(m["content"])
            
            prompt = st.chat_input("Type a message...")
            c_voice, c_reset = st.columns([0.2, 0.8])
            with c_voice:
                audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key='recorder')
            with c_reset:
                if st.button("Clear Chat", width="stretch", type="primary", icon=":material/delete_sweep:"):
                    clear_chat_history(subject)
                    # This resets the voice recording state
                    st.session_state.last_audio = None 
                    st.toast(f"Chat history for {subject} cleared!")
                    st.rerun() # This forces the page to reload and call load_chat_history()

            final_input = None
            if prompt: final_input = prompt
            elif audio:
                if audio['bytes'] != st.session_state.last_audio:
                    st.session_state.last_audio = audio['bytes']
                    with st.spinner("Transcribing..."):
                        text = transcribe_audio(audio['bytes'])
                        if text: final_input = text
            
            if final_input:
                save_message(subject, "user", final_input)
                with chat_con:
                    with st.chat_message("user"): st.markdown(final_input)
                
                # --- NEW AGENTIC LOGIC ---
                if subject == "General Chat":
                    ans = ChatGroq(model="llama-3.1-8b-instant").invoke(final_input).content
                else:
                    # Add a status indicator so the user sees the "Agent" thinking
                    with st.status("Agent working...", expanded=True) as status:
                        st.write("🔍 Retrieving documents...")
                        # The function call is now self-contained
                        from agent_tools import agentic_rag_response
                        ans = agentic_rag_response(subject, final_input)
                        
                        # Interpret the result for the UI
                        if "I checked your notes, but" in ans:
                            status.update(label="❌ No relevant notes found", state="error", expanded=False)
                        elif "**Warning" in ans:
                            status.update(label="⚠️ Fact-check warning", state="error", expanded=False)
                        else:
                            status.update(label="✅ Verified Answer Generated", state="complete", expanded=False)
                
                save_message(subject, "assistant", ans)
                st.rerun()

        # TAB 2: QUIZ
        elif tabs == "Quiz":
            if subject == "General Chat": st.warning("Open a course.")
            else:
                if st.button("New Quiz", width="stretch", type="primary", icon=":material/casino:"):
                    with st.spinner("Generating..."):
                        st.session_state.quiz_data[subject] = generate_quiz(subject)
                        st.rerun()
                
                if subject in st.session_state.quiz_data:
                    quiz = st.session_state.quiz_data[subject]
                    if not quiz: st.error("Not enough text found in PDFs to generate questions.")
                    else:
                        # We use a form so the UI doesn't refresh every time you click a radio button
                        with st.form("quiz_form"):
                            user_answers = []
                            for i, q in enumerate(quiz):
                                st.markdown(f"### Q{i+1}: {q['question']}")
                                choice = st.radio("Select an answer:", q['options'], key=f"quiz_opt_{i}")
                                user_answers.append(choice)
                                st.divider()
                            
                            submitted = st.form_submit_button("Submit Answers", type="primary", icon=":material/check_circle:")
                            
                        if submitted:
                            score = 0
                            for i, q in enumerate(quiz):
                                is_correct = user_answers[i] == q['answer']
                                if is_correct:
                                    score += 1
                                    st.success(f"**Question {i+1}: Correct!**")
                                else:
                                    st.error(f"**Question {i+1}: Incorrect.**")
                                    st.info(f"**Correct Answer:** {q['answer']}")
                                
                                # --- THE FIX: Display the Explanation ---
                                st.write(f"*Explanation:* {q['explanation']}")
                                st.divider()
                            
                            st.metric("Final Score", f"{score}/{len(quiz)}")

        # TAB 3: SUMMARIES
        elif tabs == "Summaries":
            if subject == "General Chat": 
                st.warning("Open a course.")
            else:
                if st.button("Generate Summary", width="stretch", type="primary", icon=":material/bolt:"):
                    with st.spinner("Reading..."):
                        st.session_state.summaries[subject] = generate_summary(subject)
                        st.rerun()
                
                if subject in st.session_state.summaries:
                    summary_text = st.session_state.summaries[subject]
                    st.markdown(summary_text)
                    
                    st.divider()
                    
                    # --- NEW DOWNLOAD BUTTON ---
                    from agent_tools import create_pdf_bytes
                    
                    pdf_data = create_pdf_bytes(summary_text, title=f"Summary: {subject}")
                    
                    st.download_button(
                        label="Download as PDF",
                        data=pdf_data,
                        file_name=f"{subject}_Summary.pdf",
                        mime="application/pdf",
                        icon=":material/download:",
                        use_container_width=True
                    )

        # TAB 4: FLASHCARDS
        # TAB 4: FLASHCARDS
        elif tabs == "Flashcards":
            st.subheader(":material/style: Flashcards")
            if subject == "General Chat": 
                st.warning("Open a course to use flashcards.")
            else:
                # Add a clear check if documents actually exist
                files = list_files(subject)
                if not files:
                    st.info("Please upload some PDFs first to generate flashcards.")
                else:
                    if st.button("Generate Deck", type="primary"):
                        with st.spinner("Creating cards..."):
                            # Clear old cards first to ensure the UI sees a change
                            st.session_state.flashcards[subject] = [] 
                            
                            cards = generate_flashcards(subject)
                            if cards:
                                st.session_state.flashcards[subject] = cards
                                st.session_state.flipped = {i: False for i in range(len(cards))}
                                st.rerun() # Force immediate UI update
                            else:
                                st.error("Could not find enough content. Try uploading another PDF.")
                    
                    deck = st.session_state.flashcards.get(subject, [])
                    if deck:
                        st.divider()
                        # Use columns to make them look like a stack or a grid
                        for i, card in enumerate(deck):
                            is_flipped = st.session_state.flipped.get(i, False)
                            # Show "Back" if flipped, else "Front"
                            content = card["back"] if is_flipped else card["front"]
                            sub_label = "**DEFINITION** (Click to see term)" if is_flipped else "**TERM** (Click to see definition)"
                            
                            if st.button(f"{content}\n\n{sub_label}", key=f"card_{i}_{subject}", width="stretch"):
                                st.session_state.flipped[i] = not is_flipped
                                st.rerun()
                    else:
                        st.info("No flashcards yet. Click 'Generate Deck'.")