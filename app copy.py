import streamlit as st
import os
import sqlite3
import pandas as pd
import json
import base64
import gc
import time
import shutil
from typing import List
from sqlalchemy import create_engine
from dotenv import load_dotenv
from openai import OpenAI as OfficialOpenAI

from graph import build_graph, AgentState
from sql_calls import generate_sql, execute_sql, generate_final_answer
from utils import (
    drop_all_tables,
    upload_multiple_excels_to_sqlite,
)
from vectors import (
    build_dual_retriever_system,
)

# Load OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OfficialOpenAI(api_key=os.environ["OPENAI_API_KEY"])

SQLITE_DB = "uploaded_my_chat_.db"
INDEX_DIR = "index_storage"

st.session_state["engine"] = create_engine(f"sqlite:///{SQLITE_DB}", echo=False)

def delete_old_indexes(engine):
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
    if "engine" not in st.session_state:
        st.session_state["engine"] = create_engine(f"sqlite:///{SQLITE_DB}", echo=False)

    engine = st.session_state["engine"]
    st.set_page_config(page_title="Semantic SQL Chat", layout="wide")

    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.table_infos = []
        st.session_state.chat_history = []
        st.session_state.schema_retriever = None
        st.session_state.table_info_retriever = None
        st.session_state.agent = build_graph()

    left, right = st.columns([0.2, 0.8])

    # -------- LEFT PANEL: Upload Excel --------
    with left:
        st.header("📁 Upload Excel")
        files = st.file_uploader("Upload Excel or CSV files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

        if files and not st.session_state.table_info_retriever:
            delete_old_indexes(engine)
            drop_all_tables(engine)

            st.session_state.table_infos = []
            st.session_state.chat_history = []

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

        # Chat styling
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

        # Chat history
        with st.container():
            for entry in st.session_state.chat_history:
                st.markdown(f"""
                    <div class='chat-container'>
                        <div class='chat-bubble'>
                            <strong>🧑 You:</strong><br>{entry['question']}
                        </div>
                        <div class='chat-bubble assistant'>
                            <strong>🤖 Assistant:</strong><br>{entry['answer']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                if entry.get("graph"):
                    st.image(base64.b64decode(entry["graph"]), caption="Generated Graph")

        # --------- New User Input ---------
        with st.form("chat_form", clear_on_submit=True):
            user_query = st.text_input("Ask Anything", key="chat_input", placeholder="e.g., Show me sales trend for bikes...")
            send = st.form_submit_button("Ask")

        if send and user_query.strip():
            with st.spinner("⏳ Thinking..."):
                agent_state = AgentState(user_query=user_query)
                result_state = st.session_state.agent.invoke(agent_state, {"recursion_limit": 35})
                result_obj = AgentState(**result_state)

                chat_entry = {
                    "question": user_query,
                    "answer": result_obj.answer or "",
                }
                if getattr(result_obj, "graph", None):
                    chat_entry["graph"] = result_obj.graph
                st.session_state.chat_history.append(chat_entry)

            st.rerun()

if __name__ == "__main__":
    main()
