import os
import json
import pandas as pd
import sqlite3
from openai import OpenAI as OfficialOpenAI
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from prompts import (
    SQL_GENERATION_PROMPT,
    FINAL_ANSWER_PROMPT,
)

# Load OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OfficialOpenAI(api_key=OPENAI_API_KEY)
llm = LlamaOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

SQLITE_DB = "uploaded_my_chat_.db"



def generate_sql(user_query, schema_text, table_info_text="", column_value_text=""):
    context = f"{schema_text}\n\n{table_info_text}\n\n{column_value_text}"
    
    prompt = SQL_GENERATION_PROMPT.format(
        context=context,
        user_query=user_query,
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

# 3. Execute SQL safely and return result or error
def execute_sql(query):
    try:
        with sqlite3.connect(SQLITE_DB) as conn:
            df = pd.read_sql_query(query, conn)
            if df.empty:
                return "No data found."
            return df
    except Exception as e:
        return f"SQL Error: {e}"

# 4. Convert result to natural language using final answer prompt
def generate_final_answer(user_question, result_df):
    if isinstance(result_df, str):
        return "The SQL query could not be completed due to a syntax or logic error."

    if result_df.empty:
        return "The query returned no results."

    table_text = result_df.to_markdown(index=False)
    prompt = FINAL_ANSWER_PROMPT.format(
        user_question=user_question,
        table_text=table_text
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
