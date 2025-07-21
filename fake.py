# ✅ Step 1: Add new prompt template
ANSWER_VALIDATOR_PROMPT = '''
You are an expert data analyst assistant.
The user asked: "{clarified_query}"
The generated answer is:
"""
{final_answer}
"""

Evaluate whether the answer is complete and covers the user's full intent.
If not, identify what is missing (e.g., missing columns, conditions, breakdowns, etc.).

Respond in JSON format:
{
  "is_answer_complete": true/false,
  "missing_info": "Description of what is missing."
}
'''

# ✅ Step 2: Add to AgentState
class AgentState(BaseModel):
    ...
    previous_answer: Optional[str] = None
    missing_info: Optional[str] = None
    retries: int = 0

# ✅ Step 3: Create the answer validation node
def validate_answer_node(state: AgentState) -> AgentState:
    prompt = ANSWER_VALIDATOR_PROMPT.format(
        clarified_query=state.clarified_query,
        final_answer=state.answer
    )
    try:
        response = llm.complete(prompt).text.strip()
        print("\n🧪 Answer Validator Response:\n", response)
        parsed = json.loads(response)

        if not parsed.get("is_answer_complete", True):
            state.previous_answer = state.answer
            state.missing_info = parsed.get("missing_info", "Missing details not specified")
            state.retries += 1
            return state
        else:
            state.previous_answer = None
            state.missing_info = None
            state.retries = 0
            return state

    except Exception as e:
        state.answer = f"⚠️ Answer validation failed: {str(e)}"
        return state

# ✅ Step 4: Modify generate_sql_node to use missing_info

def generate_sql_node(state: AgentState) -> AgentState:
    if not state.clarified_query:
        state.answer = "I couldn't understand your question. Please clarify your query first."
        return state

    schema_text, table_info_text = combine_retriever_results(
        st.session_state.schema_retriever,
        st.session_state.table_info_retriever,
        state.clarified_query,
    )
    print("\n📦 Schema Text:", schema_text)
    print("\n📊 Table Info Text:", table_info_text)

    # Use feedback or missing_info
    additions = []
    if state.feedback:
        additions.append(f"Note: Previous SQL had issues: {state.feedback}")
    if state.missing_info:
        additions.append(f"Also include missing context: {state.missing_info}")
    prompt_feedback = "\n".join(additions)

    updated_query = state.clarified_query + ("\n" + prompt_feedback if prompt_feedback else "")
    sql_code = generate_sql(updated_query, schema_text, table_info_text)
    state.sql_query = sql_code.replace("```sql", "").replace("```", "").strip()

    # Reset
    state.feedback = None
    state.missing_info = None
    return state

# ✅ Step 5: Modify graph flow
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

    # Answer validation outcome
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

# Save to file
    with open("langgraph_flow.png", "wb") as f:
        f.write(png_bytes)

    return workflow.compile()
