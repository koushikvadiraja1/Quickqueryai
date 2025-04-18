import pandas as pd
import streamlit as st
import database
import json
import os

# Initialize empty reports dictionary in session state if it doesn't exist
def initialize_reports():
    """
    Initialize the custom reports in session state and load any saved reports
    """
    if 'custom_reports' not in st.session_state:
        st.session_state.custom_reports = {}
        
        # Try to load saved reports from file
        try:
            if os.path.exists('custom_reports.json'):
                with open('custom_reports.json', 'r') as f:
                    saved_reports = json.load(f)
                    st.session_state.custom_reports = saved_reports
        except Exception as e:
            st.error(f"Error loading saved reports: {str(e)}")

# Save reports to a JSON file for persistence
def save_reports_to_file():
    """
    Save the current reports to a JSON file for persistence
    """
    try:
        with open('custom_reports.json', 'w') as f:
            json.dump(st.session_state.custom_reports, f)
    except Exception as e:
        st.error(f"Error saving reports: {str(e)}")

def add_custom_report(name, description, query):
    """
    Add a custom report to the reports dictionary in session state.
    
    Args:
        name: Report name
        description: Report description
        query: SQL query for the report
    
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Initialize reports if needed
        initialize_reports()
        
        # Validate the name
        if not name or len(name.strip()) == 0:
            return False, "Report name cannot be empty"
        
        # Validate the description
        if not description or len(description.strip()) == 0:
            return False, "Report description cannot be empty"
        
        # Validate the query
        if not query or len(query.strip()) == 0:
            return False, "SQL query cannot be empty"
        
        # Check if the report name already exists
        if name in st.session_state.custom_reports:
            return False, f"A report with name '{name}' already exists"
        
        # Add the report to session state
        st.session_state.custom_reports[name] = {
            "description": description,
            "query": query
        }
        
        # Save reports to file for persistence
        save_reports_to_file()
        
        return True, f"Report '{name}' added successfully"
    except Exception as e:
        return False, f"Error adding report: {str(e)}"

def delete_custom_report(name):
    """
    Delete a custom report from the reports dictionary.
    
    Args:
        name: Report name to delete
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Initialize reports if needed
        initialize_reports()
        
        # Check if the report exists
        if name not in st.session_state.custom_reports:
            return False, f"Report '{name}' not found"
        
        # Remove the report
        del st.session_state.custom_reports[name]
        
        # Save reports to file for persistence
        save_reports_to_file()
        
        return True, f"Report '{name}' deleted successfully"
    except Exception as e:
        return False, f"Error deleting report: {str(e)}"

def get_report_list():
    """
    Get list of all available reports.
    
    Returns:
        List of report names
    """
    # Initialize reports if needed
    initialize_reports()
    return list(st.session_state.custom_reports.keys())

def get_report_description(report_name):
    """
    Get description for a specific report.
    
    Args:
        report_name: Name of the report
    
    Returns:
        Report description
    """
    # Initialize reports if needed
    initialize_reports()
    
    if report_name in st.session_state.custom_reports:
        return st.session_state.custom_reports[report_name]["description"]
    return ""

def get_report_query(report_name):
    """
    Get SQL query for a specific report.
    
    Args:
        report_name: Name of the report
    
    Returns:
        SQL query for the report
    """
    # Initialize reports if needed
    initialize_reports()
    
    if report_name in st.session_state.custom_reports:
        return st.session_state.custom_reports[report_name]["query"]
    return ""

def execute_report(conn, report_name, create_view=False):
    """
    Execute a report and return the results.
    Optionally creates a temporary view for the report.
    
    Args:
        conn: Database connection
        report_name: Name of the report to execute
        create_view: Whether to create a temporary view for the report
    
    Returns:
        pandas DataFrame with report results or a tuple (DataFrame(), error_message) if error
        If create_view is True, also returns the name of the created view as third element
    """
    # Initialize reports if needed
    initialize_reports()
    
    if report_name not in st.session_state.custom_reports:
        return pd.DataFrame(), f"Report '{report_name}' not found.", None
    
    try:
        query = st.session_state.custom_reports[report_name]["query"]
        
        # Validate query before executing
        if not query or len(query.strip()) == 0:
            return pd.DataFrame(), "The report query is empty.", None
        
        # Try to execute the report query
        try:
            results = database.execute_sql_query(conn, query)
            
            # Create a temporary view if requested
            view_name = None
            if create_view:
                # Create a safe view name from the report name
                view_name = f"temp_view_{report_name.lower().replace(' ', '_')}"
                
                # First drop the view if it exists
                drop_view_query = f"DROP VIEW IF EXISTS {view_name}"
                try:
                    database.execute_sql_query(conn, drop_view_query)
                except:
                    # Ignore errors when dropping view
                    pass
                
                # Create the view
                # Some PostgreSQL versions or connections may not directly support TEMPORARY VIEW
                # Try alternative approach using cursor.execute directly
                try:
                    cursor = conn.cursor()
                    # Drop view if exists
                    cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
                    conn.commit()
                    
                    # Create the view
                    create_view_query = f"CREATE VIEW {view_name} AS {query}"
                    cursor.execute(create_view_query)
                    conn.commit()
                    cursor.close()
                    st.success(f"Created view '{view_name}' for report '{report_name}'")
                except Exception as e:
                    st.error(f"Failed to create view: {str(e)}")
                    view_name = None
            
            if create_view:
                return results, None, view_name
            else:
                return results, None
                
        except Exception as e:
            error_msg = str(e)
            
            # Create more specific error messages based on the type of error
            if "relation" in error_msg and "does not exist" in error_msg:
                # Extract table name from error message if possible
                import re
                table_match = re.search(r'relation "(.*?)" does not exist', error_msg)
                table_name = table_match.group(1) if table_match else "unknown"
                
                return pd.DataFrame(), f"The report references a table '{table_name}' that doesn't exist in the database. Please update the report query.", None
            elif "column" in error_msg and "does not exist" in error_msg:
                # Extract column name from error message if possible
                import re
                col_match = re.search(r'column (.*?) does not exist', error_msg)
                col_name = col_match.group(1) if col_match else "unknown"
                
                return pd.DataFrame(), f"The report references a column '{col_name}' that doesn't exist. Please update the report query.", None
            else:
                return pd.DataFrame(), f"Error executing report: {error_msg}", None
    except Exception as e:
        return pd.DataFrame(), f"Error preparing report: {str(e)}", None

def get_report_schema(report_name):
    """
    Generate a schema-like representation of a report's results.
    
    Args:
        report_name: Name of the report
    
    Returns:
        Dictionary with column information extracted from the SQL query
    """
    # Initialize reports if needed
    initialize_reports()
    
    if report_name not in st.session_state.custom_reports:
        return None
    
    query = st.session_state.custom_reports[report_name]["query"]
    
    # Parse column names from the SELECT clause
    columns = []
    try:
        # Very simple parsing - won't work for all SQL variations
        # For production use, consider using a proper SQL parser
        select_part = query.lower().split("select ")[1].split(" from ")[0]
        
        # Remove newlines and extra spaces
        select_part = " ".join(select_part.split())
        
        # Split by commas, but be careful with functions
        parts = []
        current = ""
        paren_level = 0
        
        for char in select_part:
            if char == '(':
                paren_level += 1
                current += char
            elif char == ')':
                paren_level -= 1
                current += char
            elif char == ',' and paren_level == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char
                
        if current:
            parts.append(current.strip())
            
        # Extract column names or aliases
        for part in parts:
            if " as " in part.lower():
                alias = part.lower().split(" as ")[-1].strip()
                columns.append(alias)
            else:
                # For columns without aliases, use the last part after a dot or the whole expression
                col_name = part.split(".")[-1].strip()
                columns.append(col_name)
                
        # Create schema-like structure
        schema = pd.DataFrame({
            "Column Name": columns,
            "Data Type": ["unknown"] * len(columns),
            "Nullable": ["YES"] * len(columns)
        })
        
        return schema
    except:
        # Fall back to a generic schema
        return pd.DataFrame({
            "Column Name": ["Report results"],
            "Data Type": ["various"],
            "Nullable": ["YES"]
        })