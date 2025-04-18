import pandas as pd
import streamlit as st
import time
import os
from dotenv import load_dotenv, set_key, find_dotenv

def format_sql(sql_query):
    """
    Format a SQL query for better readability.
    This is a simple implementation and could be enhanced.
    
    Args:
        sql_query: SQL query string
    
    Returns:
        Formatted SQL query
    """
    # Keywords to highlight
    keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 
                'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN', 
                'LIMIT', 'OFFSET', 'AND', 'OR', 'IN', 'NOT IN', 'EXISTS', 
                'NOT EXISTS', 'UNION', 'UNION ALL', 'INSERT INTO', 'UPDATE', 
                'DELETE FROM', 'CREATE TABLE', 'ALTER TABLE', 'DROP TABLE']
    
    formatted_query = sql_query
    
    # Add newlines after major clauses
    for keyword in keywords:
        formatted_query = formatted_query.replace(f" {keyword} ", f"\n{keyword} ")
    
    return formatted_query

def truncate_dataframe(df, max_rows=100, max_cols=20):
    """
    Truncate a DataFrame to a reasonable size for display.
    
    Args:
        df: pandas DataFrame
        max_rows: Maximum number of rows to display
        max_cols: Maximum number of columns to display
    
    Returns:
        Truncated DataFrame
    """
    if df.empty:
        return df
        
    # Truncate rows
    if len(df) > max_rows:
        df = df.head(max_rows)
        
    # Truncate columns
    if len(df.columns) > max_cols:
        df = df[df.columns[:max_cols]]
        
    return df

def get_display_name(column_name):
    """
    Convert database column names to more readable display names.
    
    Args:
        column_name: Original column name
    
    Returns:
        Formatted display name
    """
    # Replace underscores with spaces
    display_name = column_name.replace('_', ' ')
    
    # Capitalize words
    display_name = display_name.title()
    
    return display_name

def create_download_link(df, filename):
    """
    Create a download link for a DataFrame.
    
    Args:
        df: pandas DataFrame
        filename: Name for the downloaded file
    
    Returns:
        CSV data and filename
    """
    csv = df.to_csv(index=False)
    return csv, filename

def calculate_query_time(start_time):
    """
    Calculate and format query execution time.
    
    Args:
        start_time: Start time of the query
    
    Returns:
        Formatted execution time string
    """
    execution_time = time.time() - start_time
    
    if execution_time < 0.001:
        return f"{execution_time * 1000000:.2f} μs"
    elif execution_time < 1:
        return f"{execution_time * 1000:.2f} ms"
    else:
        return f"{execution_time:.2f} s"

def get_dataframe_summary(df):
    """
    Generate a summary of a DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary with DataFrame summary information
    """
    if df.empty:
        return {
            "rows": 0,
            "columns": 0,
            "memory_usage": "0 KB",
            "null_counts": {}
        }
    
    # Calculate null counts per column
    null_counts = df.isnull().sum().to_dict()
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
        "null_counts": null_counts
    }

def cache_dataframe(key, df):
    """
    Cache a DataFrame in the session state.
    
    Args:
        key: Key for the cached DataFrame
        df: pandas DataFrame to cache
    """
    st.session_state[key] = df

def get_cached_dataframe(key):
    """
    Retrieve a cached DataFrame from session state.
    
    Args:
        key: Key for the cached DataFrame
    
    Returns:
        Cached DataFrame or None if not found
    """
    return st.session_state.get(key)

def load_env_vars():
    """
    Load environment variables from .env file.
    Creates the file if it doesn't exist.
    
    Returns:
        Dictionary of environment variables loaded from .env file
    """
    # Find .env file or create it if it doesn't exist
    env_file = find_dotenv()
    if not env_file:
        with open('.env', 'w') as f:
            pass
        env_file = '.env'
    
    # Load environment variables from .env file
    load_dotenv(env_file)
    
    # Set default OLLAMA_HOST if not present
    if not os.environ.get("OLLAMA_HOST"):
        # Check if we have an ollama_host value saved in the .env file
        ollama_host = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
        os.environ["OLLAMA_HOST"] = ollama_host
    
    # Return a dictionary of all environment variables
    return dict(os.environ)

def save_api_key(provider, api_key):
    """
    Save an API key to .env file for persistence.
    
    Args:
        provider: Name of the provider (e.g., 'OPENAI', 'ANTHROPIC', 'MISTRAL', 'OLLAMA_HOST',
                 'custom_api_endpoint', 'custom_api_key')
        api_key: API key or configuration value to save
    
    Returns:
        True if successful, False otherwise
    """
    if not api_key:
        return False
    
    try:
        # Find .env file or create it if it doesn't exist
        env_file = find_dotenv()
        if not env_file:
            with open('.env', 'w') as f:
                pass
            env_file = '.env'
        
        # Handle special case for Ollama host
        if provider.lower() == "ollama_host":
            # Save as OLLAMA_HOST in .env file
            key_name = "OLLAMA_HOST"
            
            # Also save to OLLAMA_HOST_URL for backward compatibility
            set_key(env_file, "OLLAMA_HOST_URL", api_key)
            os.environ["OLLAMA_HOST_URL"] = api_key
        # Handle special case for custom API endpoint
        elif provider.lower() == "custom_api_endpoint":
            key_name = "CUSTOM_LLM_ENDPOINT"
            
            # Update session state if available
            if st and hasattr(st, 'session_state'):
                st.session_state["custom_api_endpoint"] = api_key
        # Handle special case for custom API key
        elif provider.lower() == "custom_api_key":
            key_name = "CUSTOM_LLM_API_KEY"
            
            # Update session state if available
            if st and hasattr(st, 'session_state'):
                st.session_state["custom_api_key"] = api_key
        else:
            # Format the key name for API keys (e.g., OPENAI_API_KEY)
            key_name = f"{provider.upper()}_API_KEY"
        
        # Save the key to .env file
        set_key(env_file, key_name, api_key)
        
        # Also set in environment variables for current session
        os.environ[key_name] = api_key
        
        return True
    except Exception as e:
        print(f"Error saving API key or configuration: {e}")
        return False

def get_api_key(provider):
    """
    Get API key or configuration value from environment variables or session state.
    
    Args:
        provider: Name of the provider (e.g., 'openai', 'anthropic', 'mistral', 'ollama_host',
                 'custom_api_endpoint', 'custom_api_key')
    
    Returns:
        API key/value string or None if not found
    """
    # Handle special case for Ollama host
    if provider.lower() == "ollama_host":
        # Check session state first
        if "ollama_host" in st.session_state:
            return st.session_state["ollama_host"]
        
        # Then try environment variables in order of preference
        return os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_HOST_URL")
    
    # Handle special case for custom API endpoint
    if provider.lower() == "custom_api_endpoint":
        # Check session state first
        if "custom_api_endpoint" in st.session_state:
            return st.session_state["custom_api_endpoint"]
        
        # Then try environment variables
        return os.environ.get("CUSTOM_LLM_ENDPOINT")
    
    # Handle special case for custom API key
    if provider.lower() == "custom_api_key":
        # Check session state first
        if "custom_api_key" in st.session_state:
            return st.session_state["custom_api_key"]
        
        # Then try environment variables
        return os.environ.get("CUSTOM_LLM_API_KEY")
    
    # Format the key name for API keys (e.g., OPENAI_API_KEY)
    key_name = f"{provider.upper()}_API_KEY"
    
    # Check session state first (for temporary keys)
    if key_name in st.session_state:
        return st.session_state[key_name]
    
    # Then check environment variables (for persistent keys)
    return os.environ.get(key_name)

def delete_api_key(provider):
    """
    Delete an API key or configuration value from .env file and environment variables.
    
    Args:
        provider: Name of the provider (e.g., 'OPENAI', 'ANTHROPIC', 'MISTRAL', 'OLLAMA_HOST',
                 'custom_api_endpoint', 'custom_api_key')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Find .env file
        env_file = find_dotenv()
        if not env_file:
            return False
            
        # Handle special case for Ollama host
        if provider.lower() == "ollama_host":
            # Set default host
            default_host = "http://localhost:11434"
            
            # Reset both OLLAMA_HOST and OLLAMA_HOST_URL in .env file
            set_key(env_file, "OLLAMA_HOST", default_host)
            set_key(env_file, "OLLAMA_HOST_URL", default_host)
            
            # Update environment variables for current session
            os.environ["OLLAMA_HOST"] = default_host
            os.environ["OLLAMA_HOST_URL"] = default_host
            
            # Update session state if it exists
            if "ollama_host" in st.session_state:
                st.session_state["ollama_host"] = default_host
                
            return True
        
        # Handle special case for custom API endpoint
        elif provider.lower() == "custom_api_endpoint":
            # Set empty value in .env file
            set_key(env_file, "CUSTOM_LLM_ENDPOINT", "")
            
            # Remove from environment variables for current session
            if "CUSTOM_LLM_ENDPOINT" in os.environ:
                del os.environ["CUSTOM_LLM_ENDPOINT"]
            
            # Remove from session state if it exists
            if "custom_api_endpoint" in st.session_state:
                del st.session_state["custom_api_endpoint"]
                
            return True
            
        # Handle special case for custom API key
        elif provider.lower() == "custom_api_key":
            # Set empty value in .env file
            set_key(env_file, "CUSTOM_LLM_API_KEY", "")
            
            # Remove from environment variables for current session
            if "CUSTOM_LLM_API_KEY" in os.environ:
                del os.environ["CUSTOM_LLM_API_KEY"]
            
            # Remove from session state if it exists
            if "custom_api_key" in st.session_state:
                del st.session_state["custom_api_key"]
                
            return True
        
        # Format the key name for API keys (e.g., OPENAI_API_KEY)
        else:
            key_name = f"{provider.upper()}_API_KEY"
            
            # Set an empty value for the key in .env file (effectively deleting it)
            set_key(env_file, key_name, "")
            
            # Remove from environment variables for current session
            if key_name in os.environ:
                del os.environ[key_name]
            
            # Remove from session state if it exists
            if key_name in st.session_state:
                del st.session_state[key_name]
            
            return True
    except Exception as e:
        print(f"Error deleting API key or configuration: {e}")
        return False
