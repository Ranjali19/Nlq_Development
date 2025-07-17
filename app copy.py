import streamlit as st
import os
import sqlite3
import pandas as pd
import uuid
from typing import List
from sqlalchemy import create_engine
from dotenv import load_dotenv

from graph import build_graph, AgentState
from sql_calls import generate_sql, execute_sql, generate_final_answer
from utils import (
    drop_all_tables,
    upload_multiple_excels_to_sqlite,
)
from vectors import (
    build_dual_retriever_system,
)

# -- 1. SQLite setup for message memory --
DB_PATH = "chat_memory.db"
def init_memory_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            thread_id TEXT,
            role TEXT,
            content TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
init_memory_db()

def save_message(thread_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
        (thread_id, role, content)
    )
    conn.commit()
    conn.close()

def fetch_chat_history(thread_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY ts ASC",
        (thread_id,)
    )
    history = [{"role": row[0], "content": row[1]} for row in c.fetchall()]
    conn.close()
    return history

# -- 2. Streamlit session state initialization --
if "engine" not in st.session_state:
    st.session_state["engine"] = create_engine("sqlite:///uploaded_my_chat_.db", echo=False)
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.table_infos = []
    st.session_state.schema_retriever = None
    st.session_state.table_info_retriever = None
    st.session_state.agent = build_graph()
if "pending_agent_state" not in st.session_state:
    st.session_state.pending_agent_state = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())   # Unique per session/chat

def delete_old_indexes(engine):
    import shutil, gc, time
    INDEX_DIR = "index_storage"
    if os.path.exists(INDEX_DIR):
        try:
            st.session_state.schema_retriever = None
            st.session_state.table_info_retriever = None
            try:
                engine.dispose()
            except:
                pass
            gc.collect()
            time.sleep(1)
            shutil.rmtree(INDEX_DIR)
            st.info("Old indexes deleted.")
        except Exception as e:
            st.error(f"Failed to delete old indexes: {e}")

def main():
    st.set_page_config(page_title="Semantic SQL Chat", layout="wide")
    engine = st.session_state["engine"]

    left, right = st.columns([0.2, 0.8])

    # -------- LEFT PANEL: Upload Excel --------
    with left:
        st.header("📁 Upload Excel")
        files = st.file_uploader("Upload Excel or CSV files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

        if files and not st.session_state.table_info_retriever:
            delete_old_indexes(engine)
            drop_all_tables(engine)
            st.session_state.table_infos = []

            with st.spinner("Processing files..."):
                infos = upload_multiple_excels_to_sqlite(files, engine)
                st.session_state.table_infos = infos
                schema_r, tableinfo_r, column_val_r = build_dual_retriever_system(engine, infos)
                st.session_state.schema_retriever = schema_r
                st.session_state.table_info_retriever = tableinfo_r
                st.session_state.column_value_retriever = column_val_r

            st.success("✅ Files processed!")
            for t in infos:
                with st.expander(f"{t.table_name}"):
                    st.write(t.table_summary)

    # -------- RIGHT PANEL: Chat Interface --------
    with right:
        st.markdown("<h2 style='text-align:right; color:white;'> SQL Chat</h2>", unsafe_allow_html=True)

        st.markdown("""
            <style>
                .chat-bubble {
                    background-color: #f5f5f5;
                    padding: 1rem;
                    border-radius: 10px;
                    margin-bottom: 1rem;
                    color: black;
                    font-size: 18px;
                    font-family: 'Segoe UI', sans-serif;
                }
                .chat-bubble.assistant {
                    color: white;
                    background-color: #1c1c1c;
                }
                .chat-container {
                    display: flex;
                    flex-direction: column;
                }
            </style>
        """, unsafe_allow_html=True)

        # Rebuild history from SQLite (persistent for thread_id)
        thread_id = st.session_state.thread_id
        chat_history = fetch_chat_history(thread_id)

        with st.container():
            for entry in chat_history:
                if entry["role"] == "user":
                    st.markdown(f"""
                        <div class='chat-container'>
                            <div class='chat-bubble'>
                                <strong>🧑 You:</strong><br>{entry['content']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:  # assistant
                    st.markdown(f"""
                        <div class='chat-container'>
                            <div class='chat-bubble assistant'>
                                <strong>🤖 Assistant:</strong><br>{entry['content']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        # --------- New User Input (with HITL support) ---------
        if st.session_state.pending_question:
            user_prompt = st.session_state.pending_question
        else:
            user_prompt = "Ask Anything"

        with st.form("chat_form", clear_on_submit=True):
            user_query = st.text_input(user_prompt, key="chat_input", placeholder="e.g., Show me sales trend for bikes...")
            send = st.form_submit_button("Ask")

        if send and user_query.strip():
            with st.spinner("⏳ Thinking..."):
                # Is this a clarification or new chat?
                if st.session_state.pending_agent_state is not None:
                    previous_state = st.session_state.pending_agent_state
                    hist = fetch_chat_history(thread_id)
                    hist.append({"role": "user", "content": user_query})
                    agent_state = AgentState(
                        **previous_state.model_dump(),
                        user_query=user_query,
                        chat_history=hist
                    )
                    st.session_state.pending_agent_state = None
                    st.session_state.pending_question = None
                else:
                    hist = fetch_chat_history(thread_id)
                    hist.append({"role": "user", "content": user_query})
                    agent_state = AgentState(user_query=user_query, chat_history=hist)

                # Save user turn
                save_message(thread_id, "user", user_query)

                # Run the agent graph
                result_state = st.session_state.agent.invoke(agent_state, {"recursion_limit": 35})

                # Human-in-the-loop interrupt?
                interrupt_flag = False
                if hasattr(result_state, "interrupt"):
                    # This is a returned Command from interrupt()
                    interrupt_flag = result_state.interrupt == "human"
                elif isinstance(result_state, dict) and "interrupt" in result_state:
                    interrupt_flag = result_state["interrupt"] == "human"

                if interrupt_flag:
                    clarification_msg = getattr(result_state, "clarification_question", None) or getattr(result_state, "answer", None) or "Can you clarify?"
                    st.session_state.pending_agent_state = agent_state
                    st.session_state.pending_question = clarification_msg
                    save_message(thread_id, "assistant", clarification_msg)
                else:
                    result_obj = AgentState(**result_state)
                    assistant_reply = result_obj.answer or ""
                    save_message(thread_id, "assistant", assistant_reply)

            st.rerun()

    # (Optional) Button to start a new thread/conversation
    with st.sidebar:
        if st.button("Start New Chat"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.pending_agent_state = None
            st.session_state.pending_question = None
            st.rerun()

if __name__ == "__main__":
    main()
