import os
import pandas as pd
import streamlit as st

# PostgreSQL imports
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

# MySQL imports
import mysql.connector

def get_db_connection():
    """
    Create a connection to the PostgreSQL database using environment variables.
    Returns a connection object.
    """
    try:
        # First try DATABASE_URL if available
        database_url = os.environ.get("DATABASE_URL")
        if database_url:
            conn = psycopg2.connect(database_url)
        else:
            # Otherwise use individual connection parameters
            conn = psycopg2.connect(
                host=os.environ.get("PGHOST"),
                database=os.environ.get("PGDATABASE"),
                user=os.environ.get("PGUSER"),
                password=os.environ.get("PGPASSWORD"),
                port=os.environ.get("PGPORT")
            )
        
        # Set database type in session state
        st.session_state['db_type'] = 'postgresql'
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None
        
def get_db_connection_with_params(host, port, database, user, password, db_type="postgresql", sslmode="prefer", ssl_ca=None, ssl_verify=True):
    """
    Create a connection to a database using provided parameters.
    
    Args:
        host: Database server hostname or IP
        port: Database server port
        database: Database name
        user: Database username
        password: Database password
        db_type: Database type ("postgresql" or "mysql")
        sslmode: SSL connection mode for PostgreSQL (prefer, require, disable, etc.)
        ssl_ca: Path to SSL CA certificate file for MySQL
        ssl_verify: Whether to verify SSL certificate for MySQL
    
    Returns:
        A database connection object or None if connection failed
    """
    # Validate required parameters
    missing_params = []
    if not host:
        missing_params.append("Host")
    if not database:
        missing_params.append("Database name")
    if not user:
        missing_params.append("Username")
        
    if missing_params:
        st.error(f"Missing required connection parameters: {', '.join(missing_params)}")
        return None
    
    try:
        if db_type.lower() == "postgresql":
            # Default to 'prefer' if sslmode is None
            if sslmode is None:
                sslmode = "prefer"
                
            # Create PostgreSQL connection
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                sslmode=sslmode
            )
            
            # Store database type in session state
            st.session_state['db_type'] = "postgresql"
            return conn
            
        elif db_type.lower() == "mysql":
            # Create MySQL connection with SSL if specified
            if ssl_ca and ssl_verify:
                conn = mysql.connector.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                    ssl_ca=ssl_ca,
                    ssl_verify_cert=True
                )
            elif ssl_ca:
                conn = mysql.connector.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                    ssl_ca=ssl_ca,
                    ssl_verify_cert=False
                )
            else:
                conn = mysql.connector.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password
                )
            
            # Store database type in session state
            st.session_state['db_type'] = "mysql"
            return conn
        else:
            st.error(f"Unsupported database type: {db_type}")
            return None
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def get_all_tables(conn):
    """
    Get all table names from the connected database.
    Args:
        conn: Database connection object (PostgreSQL or MySQL)
    Returns:
        list of table names
    """
    try:
        # Check database type from session state
        db_type = st.session_state.get('db_type', 'postgresql')
        
        # First, roll back any aborted transaction (PostgreSQL only)
        if db_type == 'postgresql':
            try:
                conn.rollback()
            except:
                pass
            
        cursor = conn.cursor()
        
        # Different query based on database type
        if db_type == 'mysql':
            # For MySQL, get tables from the current database
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = DATABASE()
                ORDER BY table_name;
            """)
        else:
            # For PostgreSQL, get tables from public schema
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            
        # MySQL returns tuples while PostgreSQL depends on cursor configuration
        if db_type == 'mysql':
            tables = [row[0] for row in cursor.fetchall()]
        else:
            tables = [row[0] for row in cursor.fetchall()]
        
        # Commit transaction (PostgreSQL)
        if db_type == 'postgresql':
            conn.commit()
            
        cursor.close()
        return tables
        
    except Exception as e:
        # PostgreSQL can roll back on error
        if st.session_state.get('db_type', 'postgresql') == 'postgresql':
            try:
                conn.rollback()
            except:
                pass
        st.error(f"Error fetching tables: {e}")
        return []

def get_table_schema(conn, table_name):
    """
    Get schema information for a specific table.
    Args:
        conn: Database connection object (PostgreSQL or MySQL)
        table_name: Name of the table
    Returns:
        DataFrame with column information
    """
    try:
        # Check database type from session state
        db_type = st.session_state.get('db_type', 'postgresql')
        
        # First, roll back any aborted transaction (PostgreSQL only)
        if db_type == 'postgresql':
            try:
                conn.rollback()
            except:
                pass
            
        cursor = conn.cursor()
        
        # Different query based on database type
        if db_type == 'mysql':
            # For MySQL
            query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = %s
                ORDER BY ordinal_position;
            """
            cursor.execute(query, (table_name,))
        else:
            # For PostgreSQL
            query = sql.SQL("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = {}
                ORDER BY ordinal_position;
            """).format(sql.Literal(table_name))
            cursor.execute(query)
        
        columns = cursor.fetchall()
        
        # Commit transaction (PostgreSQL)
        if db_type == 'postgresql':
            conn.commit()
            
        cursor.close()
        
        return pd.DataFrame(columns, columns=['Column Name', 'Data Type', 'Nullable'])
    except Exception as e:
        # PostgreSQL can roll back on error
        if st.session_state.get('db_type', 'postgresql') == 'postgresql':
            try:
                conn.rollback()
            except:
                pass
        st.error(f"Error fetching schema for {table_name}: {e}")
        return pd.DataFrame()

def execute_sql_query(conn, query, params=None):
    """
    Execute a SQL query and return the results as a pandas DataFrame.
    Args:
        conn: Database connection object (PostgreSQL or MySQL)
        query: SQL query string or sql.Composed object
        params: Parameters for the query (optional)
    Returns:
        pandas DataFrame with query results
    Raises:
        Exception: If there is an error executing the query
    """
    if not conn:
        raise Exception("Database connection not available")
    
    # Check database type from session state
    db_type = st.session_state.get('db_type', 'postgresql')
    
    # For PostgreSQL, handle sql.Composed objects differently than string queries
    if db_type == 'postgresql' and hasattr(query, 'as_string'):
        # This is a sql.Composed or sql.SQL object
        # We can skip the validation since these are properly constructed
        pass
    else:
        # For regular string queries, perform validation
        if not query or not query.strip():
            raise Exception("Empty query provided")
        
        # Validate that the query starts with a SQL command
        valid_starts = ["select", "with", "show", "explain"]
        if not any(query.strip().lower().startswith(start) for start in valid_starts):
            raise Exception(f"Query must start with a valid SQL command. Got: {query[:20]}...")
    
    try:
        # First, check if there's an aborted transaction and rollback if needed (PostgreSQL only)
        if db_type == 'postgresql':
            try:
                conn.rollback()
            except:
                pass
            
        # Create cursor based on database type
        if db_type == 'mysql':
            cursor = conn.cursor(dictionary=True)  # Return results as dictionaries for MySQL
        else:
            cursor = conn.cursor(cursor_factory=RealDictCursor)  # For PostgreSQL
        
        # Execute the query with or without parameters
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        # Check if the query returns results
        if cursor.description:
            if db_type == 'mysql':
                results = cursor.fetchall()
            else:
                results = cursor.fetchall()
        else:
            # For non-SELECT queries
            results = []
        
        # Commit the transaction (both MySQL and PostgreSQL need this)
        conn.commit()
        cursor.close()
        
        # Convert results to pandas DataFrame
        if results:
            df = pd.DataFrame(results)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        # Rollback the transaction on error
        try:
            conn.rollback()
        except:
            # Ignore errors from rollback itself
            pass
        # Propagate the error to be handled by the caller
        raise Exception(f"Error executing query: {e}")

def get_table_sample(conn, table_name, limit=5):
    """
    Get a sample of data from a table.
    Args:
        conn: Database connection object (PostgreSQL or MySQL)
        table_name: Name of the table
        limit: Number of rows to retrieve
    Returns:
        DataFrame with sample data
    """
    try:
        # Check database type from session state
        db_type = st.session_state.get('db_type', 'postgresql')
        
        if db_type == 'mysql':
            # For MySQL, use parameterized query
            query = f"SELECT * FROM `{table_name}` LIMIT %s"
            return execute_sql_query(conn, query, (limit,))
        else:
            # For PostgreSQL, use sql.Identifier to safely quote the table name
            query = sql.SQL("SELECT * FROM {} LIMIT {}").format(
                sql.Identifier(table_name),
                sql.Literal(limit)
            )
            return execute_sql_query(conn, query)
    except Exception as e:
        st.error(f"Error fetching sample from {table_name}: {e}")
        return pd.DataFrame()

def get_table_counts(conn, tables):
    """
    Get the row count for each table.
    Args:
        conn: Database connection object (PostgreSQL or MySQL)
        tables: List of table names
    Returns:
        Dictionary mapping table names to row counts
    """
    # Check database type from session state
    db_type = st.session_state.get('db_type', 'postgresql')
    
    # PostgreSQL-specific rollback
    if db_type == 'postgresql':
        try:
            conn.rollback()
        except:
            pass
        
    counts = {}
    cursor = conn.cursor()
    
    for table in tables:
        try:
            if db_type == 'mysql':
                # For MySQL
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            else:
                # For PostgreSQL
                query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
                cursor.execute(query)
                
            count = cursor.fetchone()[0]
            counts[table] = count
        except Exception as e:
            if db_type == 'postgresql':
                try:
                    conn.rollback()
                except:
                    pass
            st.error(f"Error counting rows in {table}: {e}")
            counts[table] = "Error"
    
    # Commit at the end
    try:
        conn.commit()
    except:
        pass
        
    cursor.close()
    return counts