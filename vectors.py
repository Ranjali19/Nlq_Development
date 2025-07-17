# vectors.py

import os
import json
from typing import List
from sqlalchemy import inspect
import pandas as pd
from llama_index.core import Settings, VectorStoreIndex, SQLDatabase
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.schema import Document
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field
from prompts import TABLE_INFO_PROMPT
from sqlalchemy import text



class TableInfo(BaseModel):
    table_name: str = Field(...)
    table_summary: str = Field(...)

def extract_schema(engine) -> List[Document]:
    inspector = inspect(engine)
    documents = []
    for name in inspector.get_table_names():
        columns = inspector.get_columns(name)
        if not columns:
            continue
        col_lines = [f"  - {col['name']} ({str(col['type'])})" for col in columns]
        documents.append(Document(
            text=f"Table: {name}\nColumns:\n" + "\n".join(col_lines),
            metadata={"table_name": name}
        ))
    return documents

def setup_table_info_program():
    
    return LLMTextCompletionProgram.from_defaults(
        output_cls=TableInfo,
        llm=LlamaOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
        prompt_template_str=TABLE_INFO_PROMPT,
    )

def generate_table_infos(dfs: List[pd.DataFrame], actual_table_names: List[str]) -> List[TableInfo]:
    program = setup_table_info_program()
    infos = []
    for df, name in zip(dfs, actual_table_names):
        df_str = df.head(10).to_csv()
        try:
            result = program(table_str=df_str)
            if isinstance(result, str):
                json_str = result.strip().replace('\n', '\\n').replace('\r', '')
                data = json.loads(json_str)
                data["table_name"] = name
                table_info = TableInfo(**data)
            else:
                result.table_name = name
                table_info = result
            infos.append(table_info)
        except Exception as e:
            print(f"Failed to parse table summary for '{name}': {e}")
    return infos

def extract_column_values_documents(engine) -> List[Document]:
    inspector = inspect(engine)
    documents = []

    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            try:
                columns = inspector.get_columns(table_name)
                for col in columns:
                    col_name = col["name"]
                    col_type = str(col["type"]).lower()

                    # Only include likely string-type columns
                    if not any(x in col_type for x in ["char", "text", "string"]):
                        continue

                    # Safe quoted identifiers
                    query = f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL LIMIT 100'
                    try:
                        result = conn.execute(text(query)).fetchall()
                        values = [str(row[0]) for row in result if isinstance(row[0], str)]
                        if values:
                            doc_text = f"Table: {table_name}\nColumn: {col_name}\nValues: {', '.join(values)}"
                            documents.append(Document(text=doc_text, metadata={"table": table_name, "column": col_name}))
                    except Exception as e:
                        print(f"Error fetching values for {table_name}.{col_name}: {e}")
            except Exception as e:
                print(f"Skipped table '{table_name}' due to error: {e}")
    return documents


def build_dual_retriever_system(engine, table_infos: List[TableInfo]):
    Settings.llm = LlamaOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    schema_docs = extract_schema(engine)
    schema_index = VectorStoreIndex.from_documents(schema_docs)
    schema_retriever = schema_index.as_retriever(similarity_top_k=3)

    sql_database = SQLDatabase(engine)
    mapping = SQLTableNodeMapping(sql_database)
    table_schemas = [SQLTableSchema(table_name=t.table_name, context_str=t.table_summary) for t in table_infos]
    obj_index = ObjectIndex.from_objects(table_schemas, mapping, VectorStoreIndex)
    table_info_retriever = obj_index.as_retriever(similarity_top_k=3)

    col_val_docs = extract_column_values_documents(engine)
    col_val_index = VectorStoreIndex.from_documents(col_val_docs)
    column_value_retriever = col_val_index.as_retriever(similarity_top_k=3)


    return schema_retriever, table_info_retriever, column_value_retriever

def combine_retriever_results(schema_retriever, table_info_retriever, column_value_retriever, query):
    schema_text = "\n\n".join([d.text for d in schema_retriever.retrieve(query)])
    table_info_text = "\n\n".join([
        f"Table: {d.table_name}\n{d.context_str}"
        for d in table_info_retriever.retrieve(query)
    ])
    column_value_text = "\n\n".join([d.text for d in column_value_retriever.retrieve(query)])
    return schema_text, table_info_text, column_value_text
