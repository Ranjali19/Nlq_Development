from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from typing import Literal, Optional, Union
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from dotenv import load_dotenv
import json
import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from agent_state import AgentState
import re
from sql_calls import (
    generate_sql,
    execute_sql,
    generate_final_answer,
)
from prompts import (
    CLARIFIER_PROMPT,
    CLARIFICATION_PROMPT,
    LLM_SQL_CHECKER_PROMPT,
    ANSWER_VALIDATOR_PROMPT
)
from vectors import combine_retriever_results

load_dotenv()
llm = LlamaOpenAI(model="gpt-4o-mini")



# 1. Clarifier Node (LLM: checks safety, clarity, and graph intent)
def clarifier(state: AgentState) -> AgentState:
    user_query = state.user_query.strip()
    prompt = CLARIFIER_PROMPT.format(user_query=user_query)

    try:
        response = llm.complete(prompt).text.strip()
        print("Raw Clarifier Response:", response)
        parsed = json.loads(response)

        state.wants_graph = parsed.get("wants_graph", False)
        state.assistant_response = parsed.get("assistant_response", "")
        state.is_greeting = parsed.get("is_greeting", False)

        if state.is_greeting:
            state.answer = state.assistant_response or "👋 Hi! How can I help you?"
            return state

        if not parsed["is_safe"]:
            state.answer = "❌ Unsafe query detected."
            state.safety_reason = parsed.get("safety_reason", "")
            state.is_safe = False
            state.needs_clarification = True
            return state

        if parsed["needs_clarification"]:
            state.answer = parsed.get("assistant_response") or parsed.get("clarification_question", "Can you clarify?")
            state.clarified_query = None
            state.needs_clarification = True
            state.is_safe = True
            return state

        # All good
        state.clarified_query = user_query
        state.is_safe = True
        state.needs_clarification = False
        state.answer = state.assistant_response or None
        return state

    except Exception as e:
        print("Clarifier fallback error:", str(e))
        # Fallback: Show the raw LLM text (even if not JSON)
        try:
            fallback_response = llm.complete(prompt).text.strip()
        except:
            fallback_response = "Sorry, something went wrong while generating a response."

        state.answer = fallback_response
        state.clarified_query = None
        state.is_safe = True
        state.needs_clarification = True
        return state


def ask_for_clarification(state: AgentState) -> AgentState:
    if state.clarification_question:
        state.answer = state.clarification_question
        return state
    prompt = CLARIFICATION_PROMPT.format(user_query=state.user_query)
    try:
        response = llm.complete(prompt).text.strip()
        state.answer = response
    except Exception:
        state.answer = "Can you please clarify your query? It's too ambiguous."
    return state

def handle_greeting(state: AgentState) -> AgentState:
    state.answer = state.assistant_response or "👋 Hello! How can I assist you with SQL today?"
    return state

# 3. Decision Maker
def decision_maker(state: AgentState) -> Literal["greeting", "clarification", "unsafe", "sql"]:
    if state.is_greeting:
        return "greeting"
    if state.needs_clarification:
        return "clarification"
    if state.is_safe is False:
        return "unsafe"
    return "sql"

# 4. Unsafe Handler
def handle_unsafe(state: AgentState) -> AgentState:
    state.answer = "❌ Unsafe query detected. Please revise your request."
    return state

# 5. SQL Pipeline Nodes
def generate_sql_node(state: AgentState) -> AgentState:
    if not state.clarified_query:
        state.answer = "I couldn't understand your question. Please clarify your query first."
        return state

    # Retrieve all three types of context: schema, table summary, and column value info
    schema_text, table_info_text, column_value_text = combine_retriever_results(
        st.session_state.schema_retriever,
        st.session_state.table_info_retriever,
        st.session_state.column_value_retriever,
        state.clarified_query,
    )

    print("\n📦 Retrieved Schema Text:\n", schema_text)
    print("\n📊 Retrieved Table Info Text:\n", table_info_text)
    print("\n🔠 Retrieved Column Value Text:\n", column_value_text)

    # Optionally include feedback in the generation
    prompt_feedback = f"\n\nNote: Previous SQL had issues: {state.feedback}" if state.feedback else ""
    updated_query = state.clarified_query + prompt_feedback

    # Now pass all three contexts to generate_sql
    sql_code = generate_sql(updated_query, schema_text, table_info_text, column_value_text)
    
    cleaned_sql = re.sub(r"```(sql)?", "", sql_code).replace("```", "").strip()
    state.sql_query = cleaned_sql

    # Reset feedback
    state.feedback = None
    return state


def execute_sql_node(state: AgentState) -> AgentState:
    result = execute_sql(state.sql_query)
    state.result = result
    print("SQL Execution Result:\n", result)
    return state

def generate_graph_node(state: AgentState) -> AgentState:
    """
    Generates a graph from the SQL result and encodes it as base64 PNG.
    Assumes result is a DataFrame-like object.
    """
    try:
        if isinstance(state.result, (dict, list)):
            df = pd.DataFrame(state.result)
        else:
            df = state.result  # Assume already a DataFrame

        # Ensure it's not empty and has at least 2 columns
        if df.empty or df.shape[1] < 2:
            state.graph = None
            state.answer = "Not enough data to generate a meaningful graph."
            return state

        # Set up the figure
        plt.figure(figsize=(10, 6))  # Wider and taller

        # Bar plot
        df.plot(kind='bar', x=df.columns[0], y=df.columns[1], legend=True)

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')

        # Improve layout
        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        # Encode image
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        state.graph = img_base64
        state.answer = "Graph generated successfully."
    except Exception as e:
        state.graph = None
        state.answer = f"Failed to generate graph: {str(e)}"
    return state

def generate_answer_node(state: AgentState) -> AgentState:
    state.answer = generate_final_answer(state.clarified_query, state.result)
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
    """
    Checks if the generated SQL query is valid.
    You can use EXPLAIN, a dry run, or a SQL parser for more robust validation.
    """
    try:
        explain_query = f"EXPLAIN {state.sql_query}"
        explain_result = execute_sql(explain_query)

        print("SQL Validation (EXPLAIN Output):\n", explain_result)
        state.answer = None  # Valid, proceed
    except Exception as e:
        state.answer = f"enerated SQL is invalid: {str(e)}"
        state.result = None
    return state

def validate_answer_node(state: AgentState) -> AgentState:
    prompt = ANSWER_VALIDATOR_PROMPT.format(
        clarified_query=state.clarified_query,
        final_answer=state.answer
    )

    try:
        response = llm.complete(prompt).text.strip()
        print("Answer Validator Raw Response:\n", response)

        # Try cleaning up response if it has triple backticks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()

        parsed = json.loads(response)

        if not parsed.get("is_answer_complete", True):
            state.previous_answer = state.answer
            state.missing_info = parsed.get("missing_info", "Missing details not specified")
            state.retries += 1
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
    workflow.add_node("clarification", ask_for_clarification)
    workflow.add_node("unsafe", handle_unsafe)
    workflow.add_node("greeting", handle_greeting)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("llm_sql_checker", llm_sql_checker_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("generate_graph", generate_graph_node)
    workflow.add_node("validate_answer", validate_answer_node)  # ✅ NEW

    workflow.set_entry_point("clarifier")

    workflow.add_conditional_edges(
        "clarifier", decision_maker, {
            "clarification": "clarification",
            "unsafe": "unsafe",
            "sql": "generate_sql",
            "greeting": "greeting"
        }
    )

    workflow.add_edge("generate_sql", "llm_sql_checker")

    workflow.add_conditional_edges(
        "llm_sql_checker",
        lambda state: "validate_sql" if not state.feedback else "retry_sql",
        {
            "validate_sql": "validate_sql",
            "retry_sql": "generate_sql"
        }
    )

    workflow.add_edge("validate_sql", "execute_sql")

    workflow.add_conditional_edges(
        "execute_sql",
        lambda state: "generate_graph" if getattr(state, "wants_graph", False) else "generate_answer",
        {
            "generate_graph": "generate_graph",
            "generate_answer": "generate_answer"
        }
    )

    workflow.add_edge("generate_graph", "validate_answer")
    workflow.add_edge("generate_answer", "validate_answer")

    workflow.add_conditional_edges(
        "validate_answer",
        lambda state: "generate_sql" if state.missing_info and state.retries < 3 else "end",
        {
            "generate_sql": "generate_sql",
            "end": END
        }
    )

    workflow.add_edge("clarification", END)
    workflow.add_edge("unsafe", END)
    workflow.add_edge("greeting", END)


    compiled_graph = workflow.compile()

    png_bytes = compiled_graph.get_graph().draw_mermaid_png()


    with open("langgraph_flow.png", "wb") as f:
        f.write(png_bytes)

    return compiled_graph