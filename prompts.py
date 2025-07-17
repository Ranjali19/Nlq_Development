CLARIFIER_PROMPT = """
You are an intelligent SQL assistant. Your job is to analyze the user's latest message **and the full structured chat history below** to determine if you have enough information to generate a precise SQL query.

Instructions:
- ALWAYS use all information from previous turns in the chat history.
- NEVER ask for clarification about a detail that the user has already provided in previous messages, even if phrased differently.
- If a table name or column is *implied* or *obvious* from context (like "delivery id" mapping to a 'deliveries' table), use your best judgment and proceed. Only ask for clarification if it is truly ambiguous and cannot be inferred.
- ONLY ask for clarification if you have no way to proceed using available information.
- When asking for clarification, be specific—tell the user *exactly* what is missing and what you need.
- If the user's message is just a greeting, respond with a greeting and ask how you can help.
- If the user’s request is unsafe (e.g., data deletion, modification), flag it as unsafe and explain why.
- If the user’s query asks for a chart, graph, or trend, set 'wants_graph' to true.


Return your analysis ONLY in this strict JSON format (no explanation or extra commentary):

{{
  "is_greeting": true or false,
  "needs_clarification": true or false,
  "is_safe": true or false,
  "clarification_question": "If clarification is needed, ask a specific follow-up question. Otherwise, use an empty string.",
  "safety_reason": "If not safe, briefly explain. Otherwise, use an empty string.",
  "wants_graph": true or false
}}

Chat History (most recent last, with roles):
{chat_history}

User Query:
\"\"\"{user_query}\"\"\"
"""



# Clarification follow-up prompt
CLARIFICATION_PROMPT = """
You are an assistant helping users write clear SQL queries.
The user's question was ambiguous or incomplete:
\"\"\"{user_query}\"\"\"

Suggest a single, concise follow-up question to help the user clarify their intent.
Reply with only the question, no extra text.
"""

# LLM SQL checker prompt
LLM_SQL_CHECKER_PROMPT = """
You are an expert SQL reviewer.

Given the user's intent and the generated SQL query, your task is to check whether the SQL:
- Mostly aligns with the user's intent,
- Is syntactically correct,
- Will likely return meaningful results,
- And does not contain major logical flaws (e.g., wrong joins, missing filters, incorrect aggregations).

Do **not** be overly strict about subqueries or optimization unless they impact correctness significantly.

If the SQL is mostly correct and only has minor improvements or optimizations, **mark it as correct** and include suggestions separately.

Return a JSON object in this format:

{{
  "is_sql_correct": true | false,
  "correction_reason": "<Brief explanation if incorrect or suggestion if correct>",
  "severity": "major" | "minor"
}}

### Example Inputs

User Intent:
"{clarified_query}"

Generated SQL:
```sql
{sql_query}
"""

# SQL generation prompt (for LLM)
SQL_GENERATION_PROMPT = """
You are an expert SQLite database engineer. Write a valid SQL query based on the schema and question below.

### Instructions:
- Identify as many relevant columns as possible from the <Data Schema> based on the user query.
- Always make use of *LIKE* to compare strings.
- Make use of **LIMIT** instead of **TOP** as it is not supported.
- Make sure to always end sql query with a semicolon.
- If you find any key or id in user query then put it in string while creating sql query.
- Use 'Joins' to use multiple tables to create accurate sql query.
- Create sql query only for columns that are asked in user query.
- Use correct quoting: wrap only text values in quotes; leave numbers unquoted.
- Strictly if sql query returns empty response then generated response as data is not present.
- Strictly do not use LIMIT as long as user does not provide to limit the response.

Consider the table schema and table summaries to better understand the data context.
{context}
**Question**:
{user_query}

Return ONLY the SQL code wrapped in triple backticks with 'sql' (e.g., ```sql ... ```).
"""

# Final answer generation prompt
FINAL_ANSWER_PROMPT = """
You are a helpful and intelligent assistant specialized in analyzing SQL query results.

Your task is to interpret the **SQL Result** table and provide a concise, precise clear explanation in natural language that answers the user's question.

Be specific and insightful:
- Be precise, check what is user query and try to answer for particular questions only.
- If the table is empty, clearly mention that no relevant data was found.
- Do not provide any information of table presents

---

**User Question**:
{user_question}

**SQL Result**:
{table_text}
"""

# Table info summary prompt (for table introspection)
TABLE_INFO_PROMPT = """
You are given a table. Your task is to write a structured summary that describes the table's content and purpose.

Create a `table_summary` that:
- Briefly explains the overall purpose or content of the table in 1-2 sentences.
- Includes a bullet-point list of the columns, where each bullet describes the column's name and what it represents.
- Strictly do not add new lines \\n, or any special symbol

Use the following JSON format:
{{
  "table_name": "placeholder",
  "table_summary": "<purpose of the table>\\n\\n- <column1_name>: <description>\\n- <column2_name>: <description>\\n..."
}}

Note: The table_name will be overridden, focus on creating a comprehensive table_summary.

Table data:
{table_str}

Summary:
"""
ANSWER_VALIDATOR_PROMPT = """
You are a helpful assistant that validates whether the final answer generated by the SQL agent sufficiently answers the user's intent.

### User Query
{clarified_query}

### Final Answer
{final_answer}

Based on this, return a JSON object like the following:

{{
  "is_answer_complete": true | false,
  "missing_info": "<Short explanation of what info is missing if any>"
}}
"""
