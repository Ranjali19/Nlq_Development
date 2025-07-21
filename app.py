import streamlit as st
import os
import gc
import time
import shutil
import uuid
from sqlalchemy import create_engine
from dotenv import load_dotenv

from graph import build_graph, AgentState
from utils import (
    drop_all_tables,
    upload_multiple_excels_to_sqlite,
)
from vectors import (
    build_dual_retriever_system,
)
from langgraph.types import Command

load_dotenv()
SQLITE_DB = "uploaded_my_chat_.db"
INDEX_DIR = "index_storage"

if "engine" not in st.session_state:
    st.session_state["engine"] = create_engine(f"sqlite:///{SQLITE_DB}", echo=False)
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

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

def reset_chat():
    st.session_state.chat_history = []
    st.session_state.last_state = None
    st.session_state.last_interrupt = None
    st.session_state.resume_token = None
    st.session_state.last_ambiguous_query = None

def set_last_graph(result):
    """Robustly extract a graph from any result object or dict."""
    graph_b64 = None
    # Try attribute (if result is an object)
    if hasattr(result, "graph"):
        graph_b64 = getattr(result, "graph")
    # Try dictionary (if result is a dict)
    elif isinstance(result, dict) and "graph" in result:
        graph_b64 = result["graph"]
    # Some agent outputs wrap state in 'state' or similar
    elif isinstance(result, dict) and "state" in result and "graph" in result["state"]:
        graph_b64 = result["state"]["graph"]

    if graph_b64:
        st.session_state["last_graph"] = graph_b64
    else:
        st.session_state["last_graph"] = None

    # Print debug to confirm
    print("[set_last_graph] Updated last_graph, found:", bool(graph_b64), "| Length:", len(graph_b64) if graph_b64 else 0)


def add_to_history(role, content):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "content": content})

def detect_interrupt(result):
    if "__interrupt__" in result and result["__interrupt__"]:
        interrupt_obj = result["__interrupt__"][0]
        interrupt_info = interrupt_obj.value
        resume_token = interrupt_obj
        return interrupt_info, resume_token
    return None, None

def main():
    st.set_page_config(page_title="Semantic SQL Chat", layout="wide")

    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.table_infos = []
        st.session_state.chat_history = []
        st.session_state.schema_retriever = None
        st.session_state.table_info_retriever = None
        st.session_state.column_value_retriever = None
        st.session_state.agent = build_graph()
        st.session_state.last_state = None
        st.session_state.last_interrupt = None
        st.session_state.resume_token = None
        st.session_state.last_ambiguous_query = None

    engine = st.session_state["engine"]

    left, right = st.columns([0.23, 0.77])

    with left:
        st.header("📁 Upload Excel")
        files = st.file_uploader("Upload Excel or CSV files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

        if files and not st.session_state.table_info_retriever:
            delete_old_indexes(engine)
            drop_all_tables(engine)
            st.session_state.table_infos = []
            reset_chat()

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

        if st.button("🔄 Reset Chat"):
            reset_chat()
            st.rerun()

    with right:
        st.markdown("<h2 style='text-align:right; color:white;'> SQL Chat</h2>", unsafe_allow_html=True)

        # --- Chat Display ---
        for entry in st.session_state.chat_history:
            if entry["role"] == "user":
                st.markdown(f"""
                    <div style="background:#1976d2;padding:1rem;border-radius:10px;margin-bottom:1rem;color:white;">
                        <strong>🧑 You:</strong><br>{entry['content']}
                    </div>
                """, unsafe_allow_html=True)
            elif entry["role"] == "assistant":
                if entry['content'].strip().lower() not in ["graph generated successfully.", ""]:
                    st.markdown(f"""
                        <div style="background:#1c1c1c;padding:1rem;border-radius:10px;margin-bottom:1rem;color:white">
                            <strong>🤖 Assistant:</strong><br>{entry['content']}
                        </div>
                    """, unsafe_allow_html=True)

        # --- Always Display Graph if Available ---
        graph_b64 = st.session_state.get("last_graph", None)
        if graph_b64:
            print("GRAPH DEBUG:", True, len(graph_b64))
            st.image("data:image/png;base64," + graph_b64, caption="Generated Graph")
        else:
            print("GRAPH DEBUG:", False, 0)

        # --- Continue with all your HITL / clarification and chat forms as before ---
        interrupt_info = st.session_state.get("last_interrupt", None)
        resume_token = st.session_state.get("resume_token", None)
        agent = st.session_state.agent

        # --- HITL flow (unchanged except graph saving cleaned up) ---
        if interrupt_info is not None and resume_token is not None:
            human_message = interrupt_info.get("message", "Human assistance required.")
            hitl_sql = interrupt_info.get("sql", None)

            if not st.session_state.chat_history or \
                st.session_state.chat_history[-1].get("role") != "assistant" or \
                st.session_state.chat_history[-1].get("content") != human_message:
                add_to_history("assistant", human_message)

            st.info(f"🤖 Assistant needs your input: {human_message}")

            if hitl_sql is not None:
                with st.form("human_review_form", clear_on_submit=True):
                    sql_to_edit = st.text_area("Review/Edit SQL", value=hitl_sql)
                    feedback = st.text_area("Feedback (optional)")
                    submit_sql = st.form_submit_button("Approve and Continue")
                if submit_sql:
                    add_to_history("user", f"[SQL Approved/Edited]\n{sql_to_edit}\n{feedback}")
                    cmd = Command(resume={"data": sql_to_edit})
                    with st.spinner("⏳ Resuming agent..."):
                        result = agent.invoke(cmd, config={"configurable": {"thread_id": st.session_state["thread_id"]}})
                    next_interrupt, next_token = detect_interrupt(result)
                    st.session_state.last_state = result
                    set_last_graph(result)  # << KEY: always update the graph!
                    if next_interrupt:
                        st.session_state.last_interrupt = next_interrupt
                        st.session_state.resume_token = next_token
                        st.rerun()
                    else:
                        if result.get("answer", ""):
                            add_to_history("assistant", result.get("answer", ""))
                        st.session_state.last_interrupt = None
                        st.session_state.resume_token = None
                        st.session_state.last_ambiguous_query = None
                        st.rerun()
            else:
                with st.form("clarification_form", clear_on_submit=True):
                    clarification = st.text_input("Your clarification or response", "")
                    submit_clarify = st.form_submit_button("Submit")
                if submit_clarify and clarification.strip():
                    add_to_history("user", clarification)
                    if st.session_state.get("last_ambiguous_query"):
                        combined_query = f"{st.session_state['last_ambiguous_query']} {clarification}"
                    else:
                        combined_query = clarification
                    agent_state = AgentState(
                        user_query=combined_query,
                        chat_history=st.session_state.chat_history,
                        last_ambiguous_query=None
                    )
                    with st.spinner("⏳ Resuming agent..."):
                        cmd = Command(resume=agent_state)
                        result = agent.invoke(cmd, config={"configurable": {"thread_id": st.session_state["thread_id"]}})
                    next_interrupt, next_token = detect_interrupt(result)
                    st.session_state.last_state = result
                    set_last_graph(result)  # << KEY: always update the graph!
                    if next_interrupt:
                        st.session_state.last_interrupt = next_interrupt
                        st.session_state.resume_token = next_token
                        st.session_state.last_ambiguous_query = st.session_state.get("last_ambiguous_query")
                        st.rerun()
                    else:
                        if result.get("answer", ""):
                            add_to_history("assistant", result.get("answer", ""))
                        st.session_state.last_interrupt = None
                        st.session_state.resume_token = None
                        st.session_state.last_ambiguous_query = None
                        st.rerun()
        # --- Normal chat flow ---
        else:
            with st.form("chat_form", clear_on_submit=True):
                user_query = st.text_input("Ask Anything", key="chat_input", placeholder="e.g., Show me sales trend for bikes...")
                send = st.form_submit_button("Ask")

            if send and user_query.strip():
                add_to_history("user", user_query)
                agent_state = AgentState(
                    user_query=user_query,
                    chat_history=st.session_state.chat_history,
                    last_ambiguous_query=None
                )
                with st.spinner("⏳ Thinking..."):
                    result = agent.invoke(agent_state, config={"configurable": {"thread_id": st.session_state["thread_id"]}})
                interrupt_info, resume_token = detect_interrupt(result)
                st.session_state.last_state = result
                set_last_graph(result)  # << KEY: always update the graph!
                if interrupt_info:
                    st.session_state.last_interrupt = interrupt_info
                    st.session_state.resume_token = resume_token
                    st.session_state.last_ambiguous_query = user_query
                    st.rerun()
                else:
                    if result.get("answer", ""):
                        add_to_history("assistant", result.get("answer", ""))
                    st.session_state.last_interrupt = None
                    st.session_state.resume_token = None
                    st.session_state.last_ambiguous_query = None
                    st.rerun()

if __name__ == "__main__":
    main()  
