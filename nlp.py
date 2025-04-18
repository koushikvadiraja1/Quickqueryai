import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

MODEL = "gpt-4o"

load_dotenv()  # Load environment variables from .env file



# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def natural_language_to_sql(query, tables_info, selected_tables=None, db_type="postgresql"):
    """
    Convert natural language query to SQL using OpenAI API.
    
    Args:
        query: Natural language query from the user
        tables_info: Dictionary containing table schema information
        selected_tables: List of tables selected by the user (optional)
        db_type: The database type ("postgresql" or "mysql")
    
    Returns:
        Generated SQL query
    """
    # Get database type from session state if available
    if 'conn' in st.session_state and hasattr(st.session_state.conn, '_db_type'):
        db_type = getattr(st.session_state.conn, '_db_type', 'postgresql')
    
    # Prepare tables information for context
    tables_context = ""
    for table_name, schema in tables_info.items():
        if selected_tables and table_name not in selected_tables:
            continue
            
        tables_context += f"Table: {table_name}\n"
        tables_context += "Columns:\n"
        
        # Handle different schema formats
        if "column_name" in schema.columns:
            column_name_key = "column_name"
            data_type_key = "data_type"
        elif "Column Name" in schema.columns:
            column_name_key = "Column Name"
            data_type_key = "Data Type"
        else:
            # Handle any other format by inspecting what columns are available
            available_columns = schema.columns.tolist()
            if available_columns:
                st.warning(f"Unknown schema format for {table_name}. Using first column: {available_columns[0]}")
                column_name_key = available_columns[0]
                data_type_key = available_columns[1] if len(available_columns) > 1 else available_columns[0]
            else:
                st.error(f"No columns found in schema for {table_name}")
                continue
        
        for idx, row in schema.iterrows():
            try:
                col_name = row[column_name_key]
                data_type = row[data_type_key]
                tables_context += f"- {col_name} ({data_type})\n"
            except Exception as e:
                st.error(f"Error processing column in {table_name}: {str(e)}")
                continue
        
        tables_context += "\n"
    
    # Determine the proper column and table quoting for the database type
    if db_type.lower() == "mysql":
        quote_char = "`"  # MySQL uses backticks for quoting identifiers
        db_name = "MySQL"
    else:
        quote_char = "\""  # PostgreSQL uses double quotes for case-sensitive identifiers
        db_name = "PostgreSQL"
    
    # Create the system message with context
    system_message = f"""
    You are a {db_name} expert. Your task is to convert natural language queries into valid SQL queries.
    
    Here is the database schema information:
    {tables_context}
    
    Please follow these guidelines:
    1. Generate only the SQL query, nothing else
    2. Ensure the query is valid {db_name} syntax
    3. Use {quote_char} quotes around all table and column names to preserve case sensitivity
       For example: SELECT {quote_char}Column{quote_char} FROM {quote_char}Table{quote_char}
    4. Use table and column names exactly as they appear in the schema with exact case matching
    5. IMPORTANT: Use semantic matching for columns - if a user refers to a column using slightly different terminology,
       you should identify the most appropriate matching column in the schema based on semantic similarity.
       For example, if a user asks about "customer name" but the schema has "client_name", use the "client_name" column.
    6. If the user's terminology doesn't match any column name exactly, look for synonyms, related concepts, or columns
       that would likely contain the information the user is looking for.
    7. Always prioritize columns whose names or purposes are most semantically similar to what the user is asking about.
    8. If a query is ambiguous, make reasonable assumptions and choose the most likely interpretation
    9. For aggregate queries, include appropriate GROUP BY clauses
    10. Use JOINs when necessary to query across multiple tables
    11. For queries requesting a "top N" or "bottom N", use ORDER BY with LIMIT
    12. Include appropriate WHERE clauses for filtering
    
    Respond with ONLY the SQL query, no explanations.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Convert this question to SQL: {query}"}
            ],
            temperature=0,
            max_tokens=500
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Clean the SQL query - remove any markdown formatting or SQL keywords
        if sql_query.lower().startswith("```sql"):
            # Extract SQL from markdown code block
            sql_query = sql_query.split("```")[1]
            if sql_query.lower().startswith("sql"):
                sql_query = sql_query[3:].strip()
        elif sql_query.lower().startswith("sql"):
            sql_query = sql_query[3:].strip()
        
        # Ensure the query doesn't start with unexpected characters
        sql_query = sql_query.strip()
        
        return sql_query
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return None

def extract_query_metadata(query, sql, tables_info):
    """
    Extract metadata about the query to help with visualization suggestions.
    
    Args:
        query: Original natural language query
        sql: Generated SQL query
        tables_info: Dictionary containing table schema information
    
    Returns:
        Dictionary with query metadata (query type, columns, etc.)
    """
    system_message = f"""
    Analyze the natural language query and the corresponding SQL query to extract metadata.
    Respond in JSON format with the following fields:
    
    1. "query_type": one of ["aggregate", "time_series", "comparison", "distribution", "correlation", "simple"]
    2. "visualization_type": recommended visualization type like "bar", "line", "pie", "scatter", "table", etc.
    3. "x_axis": suggested column for x-axis if applicable
    4. "y_axis": suggested column for y-axis if applicable
    5. "group_by": suggested column for grouping if applicable
    6. "aggregation_function": e.g., "count", "sum", "average", etc.
    7. "time_component": boolean indicating if this is time-based
    8. "categorical_component": boolean indicating if this involves categorical data
    9. "numerical_component": boolean indicating if this involves numerical analysis
    
    Base your analysis on both the natural language query and the SQL query.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Natural language query: {query}\nSQL query: {sql}"}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        metadata = json.loads(response.choices[0].message.content)
        return metadata
    except Exception as e:
        st.error(f"Error extracting query metadata: {e}")
        return {
            "query_type": "simple",
            "visualization_type": "table",
            "time_component": False,
            "categorical_component": False,
            "numerical_component": False
        }

def explain_query(sql):
    """
    Generate a human-readable explanation of what the SQL query does.
    
    Args:
        sql: SQL query string
    
    Returns:
        Human-readable explanation
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a SQL expert who explains SQL queries in simple terms. Be concise but clear."},
                {"role": "user", "content": f"Explain what this SQL query does in simple terms:\n\n{sql}"}
            ],
            temperature=0.7,
            max_tokens=250
        )
        
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        st.error(f"Error generating query explanation: {e}")
        return "Could not generate explanation."

def suggest_improvements(sql, results=None):
    """
    Suggest improvements to the SQL query based on the query itself and optionally the results.
    
    Args:
        sql: SQL query string
        results: Query results DataFrame (optional)
    
    Returns:
        List of improvement suggestions
    """
    context = f"SQL Query: {sql}\n"
    
    if results is not None:
        if results.empty:
            context += "The query returned no results."
        else:
            context += f"The query returned {len(results)} rows."
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a SQL optimization expert. Suggest 2-3 potential improvements to SQL queries."},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        suggestions = response.choices[0].message.content.strip()
        return suggestions
    except Exception as e:
        st.error(f"Error generating improvement suggestions: {e}")
        return "Could not generate suggestions."
