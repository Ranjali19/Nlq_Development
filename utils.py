import os
import shutil
import gc
import time
import re
import pandas as pd
import numpy as np
from sqlalchemy import Table, Column, MetaData, Integer, Float, String, Boolean, Text, text
import streamlit as st
from typing import List
from vectors import generate_table_infos

INDEX_DIR = "index_storage"


def sanitize_column_name(col_name: str) -> str:
    return re.sub(r"\W+", "_", str(col_name)).strip("_") or "col"

def infer_sqlalchemy_type(series: pd.Series):
    if pd.api.types.is_integer_dtype(series): return Integer
    if pd.api.types.is_float_dtype(series): return Float
    if pd.api.types.is_bool_dtype(series): return Boolean
    if pd.api.types.is_datetime64_any_dtype(series): return String
    return Text

def drop_all_tables(engine):
    try:
        meta = MetaData()
        meta.reflect(bind=engine)
        meta.drop_all(bind=engine)
    except Exception as e:
        st.error(f"Failed to drop tables: {e}")

def deduplicate_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        base = sanitize_column_name(col)
        if base not in seen:
            seen[base] = 0
            new_cols.append(base)
        else:
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")
    return new_cols

def create_table_from_dataframe(df: pd.DataFrame, table_name: str, engine, metadata_obj):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d')
    columns = [Column(col, infer_sqlalchemy_type(df[col])) for col in df.columns]
    table = Table(table_name, metadata_obj, *columns)
    metadata_obj.create_all(engine)
    with engine.begin() as conn:
        rows = df.replace({np.nan: None}).to_dict(orient="records")
        conn.execute(table.insert(), rows)

def upload_multiple_excels_to_sqlite(file_list, engine):
    metadata = MetaData()
    dfs = []
    actual_table_names = []

    for file in file_list:
        try:
            file_ext = os.path.splitext(file.name)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(file)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file)
            else:
                print(f"Unsupported file type: {file.name}")
                continue

            name = sanitize_column_name(os.path.splitext(file.name)[0])
            df.columns = deduplicate_columns(df.columns)

            with engine.begin() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))

            create_table_from_dataframe(df, name, engine, metadata)

            dfs.append(df)
            actual_table_names.append(name)

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    return generate_table_infos(dfs, actual_table_names)
