from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import json
import re

from agent_state import AgentState
from sql_calls import generate_sql, execute_sql, generate_final_answer
from prompts import (
    CLARIFIER_PROMPT,
    LLM_SQL_CHECKER_PROMPT,
    ANSWER_VALIDATOR_PROMPT
)
from vectors import combine_retriever_results
from langchain_core.tools import tool

load_dotenv()
llm = LlamaOpenAI(model="gpt-4o-mini")

@tool
def human_assistance(query: str) -> str:
    """
    Request assistance from a human (human-in-the-loop clarification).
    """
    human_response = interrupt({"message": query})
    return human_response["data"] if "data" in human_response else ""

def clarifier(state: AgentState):
    user_query = state.user_query.strip()
    # --- SLOT-FILLING (MERGE CLARIFICATION) LOGIC ---
    if (len(user_query.split()) < 6 or user_query.isdigit()) and getattr(state, "last_ambiguous_query", None):
        merged_query = f"{state.last_ambiguous_query.strip()} {user_query.strip()}"
        user_query = merged_query

    chat_history = getattr(state, "chat_history", [])
    context = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in chat_history])
    prompt = CLARIFIER_PROMPT.format(user_query=user_query, chat_history=context)
    print("CLARIFIER_PROMPT :",prompt)
    try:
        response = llm.complete(prompt).text.strip()
        print("Raw Clarifier Response:", response)
        parsed = json.loads(response)

        state.wants_graph = parsed.get("wants_graph", False)
        print("[CLARIFIER] wants_graph =", state.wants_graph)


        # --- GREETING ---
        if parsed.get("is_greeting", False):
            state.answer = parsed.get("assistant_response", "👋 Hi! How can I help you?")
            return state

        # --- UNSAFE QUERY ---
        if not parsed.get("is_safe", True):
            return interrupt({"message": parsed.get("assistant_response", "❌ Unsafe query detected.")})

        # --- NEEDS CLARIFICATION (route to HITL node) ---
        if parsed.get("needs_clarification", False):
            # Store for slot filling
            state.last_ambiguous_query = state.user_query.strip()
            state.clarification_question = (
                parsed.get("clarification_question")
                or parsed.get("assistant_response")
                or "Could you clarify?"
            )
            state.needs_clarification = True
            return state

        # --- CLARIFIED ---
        state.clarified_query = parsed.get("clarified_query") or user_query
        state.answer = parsed.get("assistant_response", "")
        state.is_safe = True
        state.needs_clarification = False
        state.last_ambiguous_query = None
        return state

    except Exception as e:
        if hasattr(e, "args") and e.args and hasattr(e.args[0], "value"):
            return e.args[0]
        print("Clarifier fallback error:", str(e))
        return interrupt({"message": "Sorry, could you clarify your question or rephrase?"})

def human_assistance_node(state: AgentState):
    # Just trigger the interruption directly, not via the @tool!
    question = getattr(state, "clarification_question", None) or getattr(state, "answer", None) or "Could you clarify your question?"
    # INTERRUPT! (this will pause the graph)
    return interrupt({"message": question})

def human_review_node(state: AgentState):
    return state

def generate_sql_node(state: AgentState) -> AgentState:
    if not getattr(state, "clarified_query", None):
        return interrupt({"message": "I couldn't understand your question. Please clarify your query first."})
    schema_text, table_info_text, column_value_text = combine_retriever_results(
        st.session_state.schema_retriever,
        st.session_state.table_info_retriever,
        st.session_state.column_value_retriever,
        state.clarified_query,
    )
    print("\n📦 Retrieved Schema Text:\n", schema_text)
    print("\n📊 Retrieved Table Info Text:\n", table_info_text)
    print("\n🔠 Retrieved Column Value Text:\n", column_value_text)
    prompt_feedback = f"\n\nNote: Previous SQL had issues: {state.feedback}" if getattr(state, "feedback", None) else ""
    updated_query = state.clarified_query + prompt_feedback
    sql_code = generate_sql(updated_query, schema_text, table_info_text, column_value_text)
    cleaned_sql = re.sub(r"```(sql)?", "", sql_code).replace("```", "").strip()
    state.sql_query = cleaned_sql
    return state

def llm_sql_checker_node(state: AgentState) -> AgentState:
    prompt = LLM_SQL_CHECKER_PROMPT.format(
        clarified_query=state.clarified_query,
        sql_query=state.sql_query
    )
    try:
        response = llm.complete(prompt).text.strip()
        print("LLM SQL Check Response:\n", response)
        parsed = json.loads(response)
        if not parsed.get("is_sql_correct", True):
            state.feedback = parsed.get("correction_reason", "")
            state.answer = f"LLM check failed: {state.feedback}"
            return state
    except Exception as e:
        state.feedback = str(e)
        state.answer = f"LLM SQL check failed: {state.feedback}"
    return state

def validate_sql_node(state: AgentState) -> AgentState:
    try:
        explain_query = f"EXPLAIN {state.sql_query}"
        explain_result = execute_sql(explain_query)
        print("SQL Validation (EXPLAIN Output):\n", explain_result)
        state.answer = None
    except Exception as e:
        state.answer = f"Generated SQL is invalid: {str(e)}"
        state.result = None
    return state

def execute_sql_node(state: AgentState) -> AgentState:
    result = execute_sql(state.sql_query)
    if isinstance(result, pd.DataFrame):
        state.result = result.to_dict(orient="records")
    else:
        state.result = result
    return state

def after_execute_sql_route(state):
    route = "generate_graph" if getattr(state, "wants_graph", False) else "generate_answer"
    print("[PIPELINE] Routing after execute_sql:", route)
    return route


def generate_graph_node(state: AgentState) -> AgentState:
    print("DEBUG - generate_graph_node CALLED")
    try:
        if isinstance(state.result, (dict, list)):
            df = pd.DataFrame(state.result)
        else:
            df = state.result
        print("DEBUG - DataFrame columns:", df.columns)
        print("DEBUG - DataFrame head:\n", df.head())
        if not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] < 2:
            state.graph = None
            state.answer = (
                "Not enough data to generate a meaningful graph.\n"
                f"Data received:\n{df.head(3)}"
            )
            return state
        # Auto-detect likely columns if possible
        x_col, y_col = df.columns[0], df.columns[1]
        if 'count' in df.columns[1].lower():
            y_col = df.columns[1]
        plt.figure(figsize=(6, 4))
        df.plot(kind='bar', x=x_col, y=y_col, legend=True)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        state.graph = img_base64
        state.answer = "Graph generated successfully."
        print("DEBUG - state.graph type:", type(state.graph), "length:", len(state.graph) if state.graph else None)
    except Exception as e:
        state.graph = None
        state.answer = f"Failed to generate graph: {str(e)}"
    return state


def generate_answer_node(state: AgentState) -> AgentState:
    result = state.result
    if isinstance(result, list):
        result_df = pd.DataFrame(result)
    elif isinstance(result, dict):
        result_df = pd.DataFrame([result])
    else:
        result_df = result
    state.answer = generate_final_answer(state.clarified_query, result_df)
    return state

def validate_answer_node(state: AgentState) -> AgentState:
    # ----- 1. Check for graph cases first! -----
    if getattr(state, "wants_graph", False):
        # Accept either base64 string or any non-empty object
        if getattr(state, "graph", None):
            # If graph is present, consider answer complete
            state.previous_answer = None
            state.missing_info = None
            state.retries = 0
            return state
        # If graph is wanted but missing, set missing_info
        state.previous_answer = state.answer
        state.missing_info = "Graph was requested, but not generated."
        state.retries = getattr(state, "retries", 0) + 1
        print(f"[Retrying] Missing info: {state.missing_info} | Retry: {state.retries}")
        return state

    # ----- 2. Normal flow for text-based answers -----
    prompt = ANSWER_VALIDATOR_PROMPT.format(
        clarified_query=state.clarified_query,
        final_answer=state.answer
    )
    try:
        response = llm.complete(prompt).text.strip()
        print("Answer Validator Raw Response:\n", response)
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
        parsed = json.loads(response)
        if not parsed.get("is_answer_complete", True):
            state.previous_answer = state.answer
            state.missing_info = parsed.get("missing_info", "Missing details not specified")
            state.retries = getattr(state, "retries", 0) + 1
            print(f"[Retrying] Missing info: {state.missing_info} | Retry: {state.retries}")
            return state
        else:
            state.previous_answer = None
            state.missing_info = None
            state.retries = 0
            return state
    except Exception as e:
        print("🔴 JSON parsing failed:", str(e))
        print("🔴 Raw response:", response)
        state.answer = "⚠️ Answer validation failed due to unexpected response format. Please try again."
        state.result = None
        return state    


def build_graph():
    memory = MemorySaver()
    workflow = StateGraph(AgentState, memory=memory)

    workflow.add_node("clarifier", clarifier)
    workflow.add_node("human_assistance", human_assistance_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("llm_sql_checker", llm_sql_checker_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("generate_graph", generate_graph_node)
    workflow.add_node("validate_answer", validate_answer_node)

    workflow.set_entry_point("clarifier")

    workflow.add_conditional_edges(
        "clarifier",
        lambda state: (
            "generate_sql" if not getattr(state, "needs_clarification", False) and getattr(state, "is_safe", True)
            else "human_assistance" if getattr(state, "needs_clarification", False)
            else END
        ),
        {
            "generate_sql": "generate_sql",
            "human_assistance": "human_assistance",
            END: END
        }
    )
    # After HITL, go back to clarifier (resumes with user's clarification)
    workflow.add_edge("human_assistance", "clarifier")

    workflow.add_edge("generate_sql", "human_review")
    workflow.add_edge("human_review", "llm_sql_checker")
    workflow.add_edge("llm_sql_checker", "validate_sql")
    workflow.add_edge("validate_sql", "execute_sql")
    workflow.add_conditional_edges(
                    "execute_sql",
                    after_execute_sql_route,
                    {
                        "generate_graph": "generate_graph",
                        "generate_answer": "generate_answer"
                    }
                )
    workflow.add_edge("generate_graph", "validate_answer")
    workflow.add_edge("generate_answer", "validate_answer")
    workflow.add_conditional_edges(
        "validate_answer",
        lambda state: "generate_sql" if getattr(state, "missing_info", None) and getattr(state, "retries", 0) < 3 else END,
        {
            "generate_sql": "generate_sql",
            END: END
        }
    )
    compiled_graph = workflow.compile(checkpointer=memory)

    png_bytes = compiled_graph.get_graph().draw_mermaid_png()
    with open("langgraph_flow.png", "wb") as f:
        f.write(png_bytes)

    return compiled_graph
