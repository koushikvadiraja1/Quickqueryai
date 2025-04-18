import os
import json
import pandas as pd
import streamlit as st
import time
import re

# Import our model clients module
import model_clients

# Try to load environment variables from .env file if dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv package not installed, but that's okay if env vars are set another way
    pass
    
# Import database functions at module level to avoid circular imports in functions
try:
    import database
except ImportError:
    # We'll handle this at runtime if needed
    pass

class Agent:
    """Base agent class with common functionality"""
    def __init__(self, name, system_prompt, provider="openai", model=None):
        """
        Initialize an agent with a name, system prompt, provider and model.
        
        Args:
            name: Name of the agent
            system_prompt: System prompt defining the agent's role
            provider: Model provider (openai, anthropic, mistral)
            model: Specific model name to use (if None, will use provider default)
        """
        self.name = name
        self.system_prompt = system_prompt
        self.provider = provider.lower()
        self.model = model
        
        # Create the appropriate model client
        try:
            self.client = model_clients.get_model_client(provider, model)
        except Exception as e:
            st.error(f"Error initializing {provider} client: {str(e)}")
            # Create a fallback to OpenAI if available
            try:
                self.provider = "openai"
                self.model = "gpt-4o"
                self.client = model_clients.get_model_client("openai", "gpt-4o")
                st.warning(f"Falling back to OpenAI gpt-4o model. Please check your API keys.")
            except Exception:
                st.error("Could not initialize any AI model client. Please check your API keys.")
                self.client = None

    def run(self, user_prompt, temperature=0.7):
        """
        Run the agent with a user prompt.
        
        Args:
            user_prompt: Prompt for the agent to respond to
            temperature: Creativity of response (0.0 to 1.0)
            
        Returns:
            Response from the agent
        """
        if not self.client:
            return "Error: No AI model client available. Please check your API keys."
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            return self.client.generate_completion(
                messages=messages,
                temperature=temperature
            )
        except Exception as e:
            return f"Error running agent: {str(e)}"

    def run_with_json_output(self, user_prompt, temperature=0.7):
        """
        Run the agent and get JSON output.
        
        Args:
            user_prompt: Prompt for the agent to respond to
            temperature: Creativity of response (0.0 to 1.0)
            
        Returns:
            Parsed JSON from agent response
        """
        if not self.client:
            return {"error": "No AI model client available. Please check your API keys."}
            
        # Add JSON formatting instruction to the system prompt
        json_system_prompt = self.system_prompt + "\n\nYour response should be in valid JSON format."
        
        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response_format = {"type": "json_object"} if self.provider == "openai" else None
            response_text = self.client.generate_completion(
                messages=messages,
                temperature=temperature,
                response_format=response_format
            )
            
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a formatted error
            return {"error": "Failed to parse JSON response from agent"}
        except Exception as e:
            return {"error": f"Error running agent: {str(e)}"}


class SchemaAnalysisAgent(Agent):
    """Agent responsible for analyzing schema and planning queries"""
    
    def __init__(self, provider="openai", model=None):
        system_prompt = """
        You are a Schema Analysis Agent specialized in analyzing database schemas and planning SQL queries.
        Your job is to:
        1. Analyze the user's natural language question
        2. Determine which tables and columns are needed to answer the question
        3. Identify relationships between tables
        4. Plan the key components of a SQL query (SELECT columns, FROM tables, JOIN conditions, WHERE filters, GROUP BY, ORDER BY)
        5. Identify any special requirements such as date ranges, aggregations, or filtering logic
        
        Provide a structured analysis that helps translate the question into an effective SQL query.
        
        You have expertise in CRM and HR/payroll database schemas, with knowledge about common column naming patterns:
        - For employee data: employee_id, employee_name, department, position, hire_date
        - For payroll data: net_monthly_pay, gross_monthly_pay, deductions, bonuses, commission
        - For CRM data: customer_id, customer_name, contact_info, sales_rep, account_status
        
        You understand that users may refer to these columns using different terminology, and you're able to make the appropriate matches.
        """
        super().__init__("Schema Analysis Agent", system_prompt, provider=provider, model=model)
    
    def analyze_question(self, question, tables_info):
        """
        Analyze a user question and determine tables/columns needed.
        
        Args:
            question: User's natural language question
            tables_info: Dictionary containing table schema information
            
        Returns:
            JSON object with analysis results
        """
        # Format the tables_info for readability
        tables_info_str = ""
        
        # Build common payroll/employee/CRM column mappings for reference
        common_column_mappings = {
            "net_monthly_pay": ["monthly_salary", "net_salary", "net_pay", "monthly_pay", "salary_net", "salary"],
            "gross_monthly_pay": ["gross_salary", "gross_pay", "monthly_gross", "salary_gross", "total_salary"],
            "salary": ["monthly_salary", "net_salary", "gross_salary", "net_pay", "gross_pay", "pay", "wage", "earnings"],
            "employee_name": ["name", "full_name", "emp_name", "staff_name", "personnel_name"],
            "employee_id": ["emp_id", "id", "staff_id", "personnel_id", "worker_id"],
            "department": ["dept", "division", "team", "unit", "section"],
            "position": ["job_title", "role", "title", "designation", "job_role"],
            "hire_date": ["join_date", "joining_date", "employment_date", "start_date"],
            "customer_name": ["client_name", "account_name", "contact_name", "name"],
            "customer_id": ["client_id", "account_id", "contact_id", "id"],
        }
        
        # Create a reverse mapping for reference
        column_synonym_to_standard = {}
        for standard, synonyms in common_column_mappings.items():
            for synonym in synonyms:
                column_synonym_to_standard[synonym.lower()] = standard
        
        # Get all available columns across all tables
        all_available_columns = []
        
        # Format the table information
        for table_name, table_schema in tables_info.items():
            tables_info_str += f"\nTable: {table_name}\nColumns:\n"
            for _, row in table_schema.iterrows():
                column_name = row.get('column_name', 'Unknown')
                data_type = row.get('data_type', 'Unknown')
                nullable = row.get('is_nullable', 'Unknown')
                tables_info_str += f"  - {column_name} ({data_type}, Nullable: {nullable})\n"
                all_available_columns.append(column_name)
        
        # Build a string with the column mappings to help the agent understand semantic relationships
        column_mapping_guidance = "Common column terminology mappings:\n"
        for standard, synonyms in common_column_mappings.items():
            # Only include mappings for columns that actually exist in the database
            if standard in all_available_columns:
                synonym_str = ", ".join(synonyms)
                column_mapping_guidance += f"- '{standard}' might be referred to as: {synonym_str}\n"
        
        prompt = f"""
        I need to analyze this question and the database schema to determine how to build a SQL query.
        
        Question: "{question}"
        
        Database Schema:
        {tables_info_str}
        
        {column_mapping_guidance}
        
        Please analyze:
        1. Which tables and specific columns are needed to answer this question?
        2. What joins will be required (if any) and on which columns?
        3. What filters (WHERE conditions) will be needed?
        4. Is any grouping or aggregation required? If so, which columns and functions?
        5. Is any sorting required? If so, on which columns and in what order?
        6. Are there any special considerations for this query?
        
        IMPORTANT: When matching user terminology to actual database columns:
        - First check for exact matches in the schema
        - Then check for semantic matches using the common terminology mappings above
        - For example, if the user asks about "net pay" or "salary", this likely refers to "net_monthly_pay" if that column exists
        - For example, if the user asks about "employee name", this likely refers to "employee_name" if that column exists
        - Be especially careful with payroll, employee, and CRM terms that might have standardized column names
        - When in doubt, prefer the exact column name from the schema that most closely matches the user's intent
        - If you identify any synonym mappings, include them in the column_mapping field
        
        Format your response as a JSON object with these keys:
        - tables: list of tables needed
        - columns: object mapping each table to list of columns needed
        - joins: list of join conditions (if any)
        - filters: list of filter conditions (if any)
        - grouping: list of columns to group by (if any)
        - aggregations: list of aggregation functions and columns (if any)
        - sorting: list of columns and directions for sorting (if any)
        - special_notes: any special considerations or explanations of semantic matches
        - column_mapping: object mapping user terms to actual column names (e.g., {{"customer name": "client_name", "net pay": "net_monthly_pay"}})
        """
        
        return self.run_with_json_output(prompt, temperature=0.1)


class SQLGenerationAgent(Agent):
    """Agent responsible for generating SQL queries"""
    
    def __init__(self, provider="openai", sql_specialized_model=None):
        system_prompt = """
        You are a SQL Generation Agent specialized in translating structured query plans into optimized SQL queries.
        Your job is to:
        1. Take a schema analysis with tables, columns, joins, and other query requirements
        2. Generate a precise, efficient SQL query based on that analysis
        3. Ensure proper handling of joins, filters, aggregations, and ordering
        4. Follow database-specific syntax and best practices
        5. Handle special cases like date conversions, string matching, and complex filtering
        6. Use semantic matching to handle inexact column references in the schema analysis
        
        You have expertise in writing error-free SQL code with these priorities:
        1. ALWAYS use exact table and column names as they appear in the schema - this is critical
        2. Check that all column names actually exist in their respective tables
        3. Ensure all table joins have the correct foreign key/primary key pairs
        4. Verify that every table used in a join is properly included in the FROM clause
        5. Validate data types match in comparison operations
        6. Add explicit type casts when necessary (e.g., ::date for dates in PostgreSQL)
        7. Use proper quoting for identifiers (double quotes for PostgreSQL, backticks for MySQL)
        8. Implement proper handling of NULL values (IS NULL/IS NOT NULL rather than =/!=)
        9. Format the query for readability with proper indentation, line breaks, and comments
        10. Double-check that syntax is correct for specified database (PostgreSQL or MySQL)
        
        Common errors to avoid:
        - Missing aliases when joining multiple tables
        - Column reference ambiguity in joins
        - Missing GROUP BY columns that are in SELECT but not aggregated
        - Division by zero possibilities
        - Using column names in ORDER BY that are aliases or expressions from SELECT
        - Invalid date format handling (use appropriate date functions)
        - Invalid casting between incompatible types
        
        Your output should be a well-formatted, executable SQL query that meets all requirements.
        """
        # Pass the provider and model to the parent class
        super().__init__("SQL Generation Agent", system_prompt, provider=provider, model=sql_specialized_model)
    
    def generate_sql(self, schema_analysis, db_type="postgresql", tables_info=None, original_question=None):
        """
        Generate SQL from the schema analysis.
        
        Args:
            schema_analysis: Analysis from the SchemaAnalysisAgent
            db_type: Database type (postgresql or mysql)
            tables_info: Dictionary containing table schema information (optional)
            original_question: The original natural language question (optional)
            
        Returns:
            JSON object with SQL query
        """
        # Extract the column_mapping from schema_analysis if available
        column_mapping = schema_analysis.get('column_mapping', {})
        column_mapping_str = ""
        if column_mapping:
            column_mapping_str = f"\nColumn mappings detected:\n{json.dumps(column_mapping, indent=2)}\n"
        
        # Check for any special notes about semantic matches
        special_notes = schema_analysis.get('special_notes', "")
        if special_notes:
            special_notes = f"\nSpecial notes from schema analysis:\n{special_notes}\n"
        
        # Format the full table schema info if provided
        tables_info_str = ""
        if tables_info:
            tables_info_str = "Full Database Schema:\n"
            for table_name, table_schema in tables_info.items():
                tables_info_str += f"\nTable: {table_name}\nColumns:\n"
                for _, row in table_schema.iterrows():
                    column_name = row.get('column_name', 'Unknown')
                    data_type = row.get('data_type', 'Unknown')
                    nullable = row.get('is_nullable', 'Unknown')
                    tables_info_str += f"  - {column_name} ({data_type}, Nullable: {nullable})\n"
            tables_info_str += "\n"
        
        # Include the original question if provided
        original_question_str = ""
        if original_question:
            original_question_str = f"Original Question: \"{original_question}\"\n\n"
        
        prompt = f"""
        {original_question_str}I need to generate a SQL query based on this schema analysis:
        
        {json.dumps(schema_analysis, indent=2)}
        
        Database type: {db_type}
        {column_mapping_str}
        {special_notes}
        {tables_info_str}
        
        Please generate a complete, executable SQL query that:
        1. Uses the identified tables and columns
        2. Implements all necessary joins
        3. Applies all required filters
        4. Handles any aggregations and grouping
        5. Sorts results as specified
        6. Follows best practices for {db_type} syntax
        
        IMPORTANT: When using columns in the SQL query:
        - Use EXACT column names as they appear in the database schema
        - If the schema analysis mapped inexact terms to actual columns (see column_mapping),
          make sure to use the actual column names in your query
        - Be aware of case sensitivity and properly quote identifiers as needed
        
        ERROR PREVENTION CHECKLIST - verify each of these before finalizing your query:
        1. ✓ Confirm all column names exactly match the schema (with correct capitalization)
        2. ✓ Verify all tables referenced in SELECT, WHERE, JOIN exist in the schema
        3. ✓ For aggregated queries, ensure all non-aggregated columns in SELECT are in GROUP BY
        4. ✓ Verify join conditions use the correct column relationships between tables
        5. ✓ Check for potential NULL issues in joins, filters, and calculations
        6. ✓ Ensure proper data type handling in comparisons (dates, numbers, strings)
        7. ✓ Add table aliases for all joined tables to prevent ambiguity
        8. ✓ For {db_type}, follow proper identifier quoting conventions
        9. ✓ Perform syntax check for missing commas, parentheses, or SQL keywords
        10. ✓ Check ORDER BY references valid columns or position numbers
        
        Some specifics to handle:
        - For PostgreSQL: Use double quotes for identifiers if needed, ensure proper date/time handling
        - For MySQL: Use backticks for identifiers if needed, handle string case sensitivity appropriately
        
        Format your response as a JSON object with these keys:
        - sql_query: the complete SQL query
        - explanation: brief explanation of key elements in the query
        - column_usage: a mapping of which schema columns were used in the query
        - error_prevention: list of specific error prevention steps you took
        """
        
        return self.run_with_json_output(prompt, temperature=0.2)


class CombinedSQLAgent(Agent):
    """Combined agent for schema analysis and SQL generation in one step"""
    
    def __init__(self, provider="openai", model=None):
        system_prompt = """
        You are a Combined SQL Generation Agent specialized in:
        1. Analyzing database schemas and user questions
        2. Identifying tables and columns needed to answer the question
        3. Generating optimized SQL queries directly
        
        You have expertise in databases, SQL, and semantic understanding to translate
        natural language queries directly to SQL using the correct column names from the schema.
        
        You have specific expertise in CRM and HR/payroll database schemas, with knowledge about common column naming patterns:
        - For employee data: employee_id, employee_name, department, position, hire_date
        - For payroll data: net_monthly_pay, gross_monthly_pay, deductions, bonuses, commission
        - For CRM data: customer_id, customer_name, contact_info, sales_rep, account_status
        
        You can handle various forms of terminology, understanding that users might refer to columns
        using different terms (for example, "net pay" or "salary" might refer to "net_monthly_pay").
        """
        super().__init__("Combined SQL Agent", system_prompt, provider=provider, model=model)
    
    def generate_sql_query(self, question, tables_info, db_type="postgresql"):
        """
        Analyze the question and generate SQL in one step.
        
        Args:
            question: User's natural language question
            tables_info: Dictionary containing table schema information
            db_type: Database type (postgresql or mysql)
            
        Returns:
            JSON object with SQL query and analysis
        """
        # Format the tables_info for readability
        tables_info_str = ""
        
        # Build common payroll/employee/CRM column mappings for reference
        common_column_mappings = {
            "net_monthly_pay": ["monthly_salary", "net_salary", "net_pay", "monthly_pay", "salary_net", "salary"],
            "gross_monthly_pay": ["gross_salary", "gross_pay", "monthly_gross", "salary_gross", "total_salary"],
            "salary": ["monthly_salary", "net_salary", "gross_salary", "net_pay", "gross_pay", "pay", "wage", "earnings"],
            "employee_name": ["name", "full_name", "emp_name", "staff_name", "personnel_name"],
            "employee_id": ["emp_id", "id", "staff_id", "personnel_id", "worker_id"],
            "department": ["dept", "division", "team", "unit", "section"],
            "position": ["job_title", "role", "title", "designation", "job_role"],
            "hire_date": ["join_date", "joining_date", "employment_date", "start_date"],
            "customer_name": ["client_name", "account_name", "contact_name", "name"],
            "customer_id": ["client_id", "account_id", "contact_id", "id"],
        }
        
        # Get all available columns across all tables
        all_available_columns = []
        
        # Format the table information
        for table_name, table_schema in tables_info.items():
            tables_info_str += f"\nTable: {table_name}\nColumns:\n"
            for _, row in table_schema.iterrows():
                column_name = row.get('column_name', 'Unknown')
                data_type = row.get('data_type', 'Unknown')
                nullable = row.get('is_nullable', 'Unknown')
                tables_info_str += f"  - {column_name} ({data_type}, Nullable: {nullable})\n"
                all_available_columns.append(column_name)
        
        # Build a string with the column mappings to help the agent understand semantic relationships
        column_mapping_guidance = "Common column terminology mappings:\n"
        for standard, synonyms in common_column_mappings.items():
            # Only include mappings for columns that actually exist in the database
            if standard in all_available_columns:
                synonym_str = ", ".join(synonyms)
                column_mapping_guidance += f"- '{standard}' might be referred to as: {synonym_str}\n"
        
        # Determine the proper column and table quoting for the database type
        if db_type.lower() == "mysql":
            quote_char = "`"  # MySQL uses backticks for quoting identifiers
            db_name = "MySQL"
        else:
            quote_char = "\""  # PostgreSQL uses double quotes for case-sensitive identifiers
            db_name = "PostgreSQL"
            
        prompt = f"""
        I need you to directly translate this natural language question into a SQL query for {db_name}.
        
        Question: "{question}"
        
        Database Schema:
        {tables_info_str}
        
        {column_mapping_guidance}
        
        Please analyze the question and generate a complete, executable SQL query that answers it.
        Follow these guidelines:
        
        1. Identify exactly which tables and columns are needed based on the question
        2. Use the EXACT column names as they appear in the schema, with proper case sensitivity
        3. Use {quote_char} quotes around table and column names to preserve case sensitivity
           For example: SELECT {quote_char}Column{quote_char} FROM {quote_char}Table{quote_char}
        4. Include all necessary JOINs, WHERE clauses, GROUP BY, and ORDER BY statements
        5. When a concept in the question doesn't match a column name exactly:
           a. First check if it matches any semantic variations in the column mapping above
           b. If not, use intelligent matching to find the most appropriate column
           c. For payroll/HR terms, be especially careful to map correctly (e.g., "salary" → "net_monthly_pay")
        
        ERROR PREVENTION CHECKLIST - verify each of these before finalizing your query:
        1. ✓ Confirm all column names exactly match the schema (with correct capitalization)
        2. ✓ Verify all tables referenced in SELECT, WHERE, JOIN exist in the schema
        3. ✓ For aggregated queries, ensure all non-aggregated columns in SELECT are in GROUP BY
        4. ✓ Verify join conditions use the correct column relationships between tables
        5. ✓ Check for potential NULL issues in joins, filters, and calculations
        6. ✓ Ensure proper data type handling in comparisons (dates, numbers, strings)
        7. ✓ Add table aliases for all joined tables to prevent ambiguity
        8. ✓ For {db_name}, follow proper identifier quoting conventions ({quote_char})
        9. ✓ Perform syntax check for missing commas, parentheses, or SQL keywords
        10. ✓ Check ORDER BY references valid columns or position numbers
        
        Format your response as a JSON object with these keys:
        - sql_query: the complete SQL query
        - explanation: brief explanation of how the query works
        - column_mappings: object showing how user terms were mapped to actual columns
        - tables_used: list of tables used in the query
        - error_prevention: list of specific error prevention steps you took
        """
        
        return self.run_with_json_output(prompt, temperature=0.1)
        
class DataAnalysisAgent(Agent):
    """Agent responsible for analyzing data and providing insights"""
    
    def __init__(self, provider="openai", model=None):
        system_prompt = """
        You are a Data Analysis Agent specialized in analyzing query results and providing insights.
        Your job is to:
        1. Analyze data returned from SQL queries
        2. Identify key patterns, trends, and outliers in the data
        3. Contextualize the results in relation to the original question
        4. Suggest visualizations that would best represent the insights
        5. Provide a clear, concise explanation of what the data shows
        
        Provide meaningful insights that help users understand their data better.
        """
        super().__init__("Data Analysis Agent", system_prompt, provider=provider, model=model)
    
    def analyze_data(self, user_question, sql_query, data, metadata=None):
        """
        Analyze query results and provide insights.
        
        Args:
            user_question: Original user question
            sql_query: SQL query used to retrieve data
            data: DataFrame with query results
            metadata: Optional metadata about the query/data
            
        Returns:
            JSON object with analysis results
        """
        # Convert DataFrame to JSON for the prompt
        try:
            data_sample = data.head(20).to_json(orient="records", date_format="iso")
        except:
            data_sample = str(data.head(20))
        
        # Get summary statistics where applicable
        try:
            numeric_stats = data.describe().to_json()
        except:
            numeric_stats = "{}"
        
        # Get column types
        column_types = {}
        for col in data.columns:
            column_types[col] = str(data[col].dtype)
        
        # Import visualization module to detect column types for better visualization suggestions
        import visualization
        detected_column_types = visualization.detect_column_types(data)
        
        # Create the prompt
        prompt = f"""
        I need to analyze these query results and provide insights.
        
        Original question: "{user_question}"
        
        SQL query used:
        ```sql
        {sql_query}
        ```
        
        Data sample (first 20 rows):
        {data_sample}
        
        Summary statistics:
        {numeric_stats}
        
        Column data types:
        {json.dumps(column_types)}
        
        Total rows: {len(data)}
        Total columns: {len(data.columns)}
        
        Detected column categories:
        Numerical columns: {detected_column_types['numerical']}
        Categorical columns: {detected_column_types['categorical']}
        Datetime columns: {detected_column_types['datetime']}
        Text columns: {detected_column_types['text']}
        
        Please provide a comprehensive analysis:
        1. What are the key insights from this data?
        2. Are there any notable patterns, trends, or outliers?
        3. How does this data answer the original question?
        4. What visualization would best represent the key insight? Be specific and provide details on which columns to use.
        5. What additional analysis might be helpful?
        
        Format your response as a JSON object with these keys:
        - insights: list of key insights (3-5 bullet points)
        - patterns: any significant patterns or trends identified
        - answer_summary: concise summary of how the data answers the original question
        - visualization_recommendations: detailed visualization specifications including:
          * viz_type: the best visualization type for this data (e.g., "bar", "line", "pie", "scatter", "heatmap", "treemap", etc.)
          * columns: a JSON object with keys representing chart elements and values being column names from the dataset
            *** VERY IMPORTANT: This MUST be a valid JSON object and not a string representation ***
            ALWAYS format the columns value as a proper JSON object, not as a string.
            For example: {{"x": "column_name1", "y": "column_name2", "color": "column_name3"}}
            Different chart types require different keys:
            - Bar/line charts: {{"x": "x_column", "y": "y_column", "color": "group_column"}}
            - Pie charts: {{"names": "category_column", "values": "numeric_column"}}
            - Scatter plots: {{"x": "x_column", "y": "y_column", "color": "group_column", "size": "size_column"}}
            - Bubble charts: {{"x": "x_column", "y": "y_column", "size": "size_column", "color": "color_column"}}
            - Heatmaps: {{"x": "x_column", "y": "y_column", "z": "z_column"}}
            - Treemap: {{"path": ["parent_column", "child_column"], "values": "size_column", "color": "color_column"}}
            - Sunburst: {{"path": ["level1_column", "level2_column"], "values": "size_column"}}
            Make sure all column names exist in the actual dataset
          * title: a descriptive title for the chart
          * description: why this visualization is appropriate for the data
          * visualization_code: Complete runnable Python code that creates this visualization using Plotly. The code should:
            1. Use Plotly Express or Plotly Graph Objects to create the visualization
            2. Use a DataFrame variable named 'df' which will be provided by the application
            3. Define a function called 'create_visualization(df)' that returns the figure
            4. Include proper imports at the top (import plotly.express as px, import plotly.graph_objects as go)
            5. Handle any necessary data transformations
            6. Use proper column names that exist in the dataset
            7. Include appropriate styling and formatting
            Example code structure:
            ```python
            import plotly.express as px
            import plotly.graph_objects as go
            
            def create_visualization(df):
                # Create the visualization
                fig = px.bar(df, x="column1", y="column2", color="column3", title="Meaningful Title")
                
                # Apply styling
                fig.update_layout(
                    template="plotly_white",
                    legend_title="Category",
                    xaxis_title="X-Axis Label",
                    yaxis_title="Y-Axis Label"
                )
                
                return fig
            ```
        - additional_analysis: suggestions for further analysis
        """
        
        return self.run_with_json_output(prompt, temperature=0.5)
    
    def explain_analysis(self, user_question, sql_query, data, metadata=None):
        """
        Provide a natural language explanation of the data analysis.
        
        Args:
            user_question: Original user question
            sql_query: SQL query used to retrieve data
            data: DataFrame with query results
            metadata: Optional metadata about the query/data
            
        Returns:
            Detailed textual explanation
        """
        # Get basic insights using the analyze_data method
        analysis = self.analyze_data(user_question, sql_query, data, metadata)
        
        # Format a more conversational explanation prompt
        prompt = f"""
        Based on the following analysis of data:
        
        {json.dumps(analysis, indent=2)}
        
        Please provide a clear, conversational explanation of what the data shows in relation to the original question:
        "{user_question}"
        
        Your explanation should:
        1. Be clear and easy to understand for non-technical users
        2. Highlight the most important insights
        3. Contextualize the findings
        4. Avoid technical jargon unless necessary
        5. Be conversational but informative
        
        Structure your response with a summary paragraph followed by key findings and their implications.
        """
        
        return self.run(prompt, temperature=0.7)
        
    def explain_analysis_streaming(self, user_question, sql_query, data, metadata=None, streamlit_placeholder=None):
        """
        Provide a natural language explanation of the data analysis with streaming output.
        
        Args:
            user_question: Original user question
            sql_query: SQL query used to retrieve data
            data: DataFrame with query results
            metadata: Optional metadata about the query/data
            streamlit_placeholder: Streamlit placeholder for streaming output
            
        Returns:
            Detailed textual explanation or None if streaming was successful
        """
        # If no placeholder provided, use the regular method
        if not streamlit_placeholder:
            return self.explain_analysis(user_question, sql_query, data, metadata)
            
        # Import here to avoid circular imports
        from model_clients import get_model_client
        
        # Get basic insights using the analyze_data method
        analysis = self.analyze_data(user_question, sql_query, data, metadata)
        
        # Format a more conversational explanation prompt
        prompt = f"""
        Based on the following analysis of data:
        
        {json.dumps(analysis, indent=2)}
        
        Please provide a clear, conversational explanation of what the data shows in relation to the original question:
        "{user_question}"
        
        Your explanation should:
        1. Be clear and easy to understand for non-technical users
        2. Highlight the most important insights
        3. Contextualize the findings
        4. Avoid technical jargon unless necessary
        5. Be conversational but informative
        
        Structure your response with a summary paragraph followed by key findings and their implications.
        """
        
        # Get model client
        client = get_model_client(self.provider, self.model)
        
        if not client:
            return "Unable to connect to model provider. Please check API key settings."
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Check if streaming is supported
        if hasattr(client, 'generate_streaming_completion'):
            try:
                # Initialize empty response
                full_response = ""
                
                # Write initial message
                streamlit_placeholder.write("Generating analysis...")
                
                # Stream response
                for text_chunk in client.generate_streaming_completion(messages, temperature=0.7):
                    # Append to the full response
                    full_response += text_chunk
                    
                    # Update the placeholder with current cumulative response
                    streamlit_placeholder.markdown(full_response)
                
                return full_response
            
            except Exception as e:
                import logging
                logging.error(f"Error in streaming explanation: {str(e)}")
                # Fall back to normal method
                return self.explain_analysis(user_question, sql_query, data, metadata)
        else:
            # Fall back to normal method if streaming not supported
            return self.explain_analysis(user_question, sql_query, data, metadata)


class SQLValidationAgent(Agent):
    """Agent responsible for validating and fixing SQL queries"""
    
    def __init__(self, provider="openai", model=None):
        system_prompt = """
        You are a SQL Validation Agent specialized in fixing and optimizing SQL queries.
        Your job is to:
        1. Check if a SQL query is syntactically valid
        2. Verify that all table and column references exist in the provided schema
        3. Fix common errors like case sensitivity, missing joins, or improper quotation
        4. Optimize the query for better performance
        5. Make sure the query actually answers the original question
        
        You specialize in identifying and fixing these common SQL errors:
        - Incorrect table or column names (including case sensitivity)
        - Missing columns in GROUP BY clauses when using non-aggregated columns in SELECT
        - Missing JOIN conditions or incorrect join relationships
        - Improper quoting of identifiers (database-specific)
        - Missing FROM clauses or tables
        - Inconsistent data types in comparisons
        - Missing aliases in joins leading to ambiguous column references
        - Improper handling of NULL values
        - Mismatched parentheses in complex expressions
        - Incorrect SQL syntax specific to PostgreSQL or MySQL
        - Improper date/time format handling
        - Potentially unsafe division operations (possible division by zero)
        - Incorrect use of aggregation functions
        - Improper subquery syntax
        - Column reference ambiguity in JOINs
        - Invalid ORDER BY references
        
        You should be extremely thorough in validating column names against the schema.
        When validating column references, consider the semantic meaning of columns:
        - If a column name in the query doesn't exactly match any schema column, try to identify the closest semantic match
        - Use intelligent fuzzy matching to correct minor discrepancies in column references
        - Consider common variations like pluralization, underscores vs. spaces, and abbreviations
        - Document any column name corrections you make in your response
        
        You have expertise in CRM and HR/payroll database schemas, with knowledge about common column naming patterns:
        - For employee data: employee_id, employee_name, department, position, hire_date
        - For payroll data: net_monthly_pay, gross_monthly_pay, deductions, bonuses, commission
        - For CRM data: customer_id, customer_name, contact_info, sales_rep, account_status
        
        You understand that users may refer to these columns using different terminology, and you're able to make the appropriate matches.
        """
        super().__init__("SQL Validation Agent", system_prompt, provider=provider, model=model)
    
    def validate_and_fix_sql(self, original_query, sql_query, tables_info, db_type="postgresql"):
        """
        Validate and fix a SQL query.
        
        Args:
            original_query: Original natural language query
            sql_query: SQL query to validate
            tables_info: Dictionary containing table schema information
            db_type: Database type (postgresql or mysql)
            
        Returns:
            JSON object with validated/fixed SQL
        """
        # Build common payroll/employee/CRM column mappings for reference
        common_column_mappings = {
            "net_monthly_pay": ["monthly_salary", "net_salary", "net_pay", "monthly_pay", "salary_net", "salary"],
            "gross_monthly_pay": ["gross_salary", "gross_pay", "monthly_gross", "salary_gross", "total_salary"],
            "salary": ["monthly_salary", "net_salary", "gross_salary", "net_pay", "gross_pay", "pay", "wage", "earnings"],
            "employee_name": ["name", "full_name", "emp_name", "staff_name", "personnel_name"],
            "employee_id": ["emp_id", "id", "staff_id", "personnel_id", "worker_id"],
            "department": ["dept", "division", "team", "unit", "section"],
            "position": ["job_title", "role", "title", "designation", "job_role"],
            "hire_date": ["join_date", "joining_date", "employment_date", "start_date"],
            "customer_name": ["client_name", "account_name", "contact_name", "name"],
            "customer_id": ["client_id", "account_id", "contact_id", "id"],
        }
        
        # Get all available columns across all tables
        all_available_columns = []
        
        # Format the tables_info for readability
        tables_info_str = ""
        for table_name, table_schema in tables_info.items():
            tables_info_str += f"\nTable: {table_name}\nColumns:\n"
            for _, row in table_schema.iterrows():
                column_name = row.get('column_name', 'Unknown')
                data_type = row.get('data_type', 'Unknown')
                nullable = row.get('is_nullable', 'Unknown')
                tables_info_str += f"  - {column_name} ({data_type}, Nullable: {nullable})\n"
                all_available_columns.append(column_name)
                
        # Build a string with the column mappings to help the agent understand semantic relationships
        column_mapping_guidance = "Common column terminology mappings:\n"
        for standard, synonyms in common_column_mappings.items():
            # Only include mappings for columns that actually exist in the database
            if standard in all_available_columns:
                synonym_str = ", ".join(synonyms)
                column_mapping_guidance += f"- '{standard}' might be referred to as: {synonym_str}\n"
        
        prompt = f"""
        I need to validate and fix this SQL query for a {db_type} database.
        
        Original question: "{original_query}"
        
        Generated SQL query:
        ```sql
        {sql_query}
        ```
        
        Database Schema:
        {tables_info_str}
        
        {column_mapping_guidance}
        
        Please verify and fix the query:
        1. Ensure all table and column names exist in the schema and are properly referenced
        2. Make sure all table name and column references use the correct case sensitivity
           - For PostgreSQL: case-sensitive unless double-quoted
           - For MySQL: case-insensitive typically but be consistent
        3. Check for missing JOIN conditions or incorrect relationships
        4. Verify that the WHERE clause properly filters according to the original question
        5. Check for other syntax errors or logical issues
        
        IMPORTANT: When validating column references, use multiple approaches:
        1. First check for exact matches in the schema
        2. Then check for semantic matches using the common terminology mappings above
        3. For example, if the query uses "net_pay" but the schema has "net_monthly_pay", replace it with "net_monthly_pay"
        4. For example, if the query uses "salary" but the schema has "net_monthly_pay", replace it with "net_monthly_pay"
        5. Be especially careful with payroll, employee, and CRM terms that might have standardized column names
        6. Look for synonyms, related concepts, or columns that would contain the same type of information
        7. Consider common variations like singular/plural forms, abbreviations, or word order differences
        
        First, analyze the query for any column references that don't exactly match the schema.
        Then, check if any of those references match the common terminology mappings above.
        If you find matches, replace the column references with the actual column names from the schema.
        
        Format your response as a JSON object with these keys:
        - is_valid: boolean indicating if the query is valid or needs fixing
        - fixed_sql: the corrected SQL query (or the original if no changes needed)
        - changes_made: list of changes made to the query (empty list if none)
        - explanation: brief explanation of what was fixed and why
        - semantic_mappings: dictionary mapping original column references to the actual schema columns they were matched with
        - is_valid: boolean indicating whether the query is now valid
        """
        
        return self.run_with_json_output(prompt, temperature=0.1)


class QueryTestingAgent(Agent):
    """Agent responsible for testing SQL queries with sample data"""
    
    def __init__(self, provider="openai", model=None):
        system_prompt = """
        You are a Query Testing Agent specialized in validating SQL queries against actual databases.
        Your job is to:
        1. Test a SQL query on a small subset of data
        2. Identify potential errors or performance issues
        3. Verify that the results match the expected output for the question
        4. Suggest modifications to improve accuracy or performance
        
        Your goal is to ensure queries are robust and produce correct results.
        """
        super().__init__("Query Testing Agent", system_prompt, provider=provider, model=model)
    
    def test_query(self, original_query, sql_query, sample_results):
        """
        Test a SQL query and verify the results.
        
        Args:
            original_query: Original natural language query
            sql_query: SQL query to test
            sample_results: DataFrame with sample results from executing the query
            
        Returns:
            JSON object with test results
        """
        # Convert DataFrame to JSON for the prompt
        try:
            results_sample = sample_results.head(10).to_json(orient="records", date_format="iso")
        except:
            results_sample = str(sample_results.head(10))
        
        prompt = f"""
        I need to verify if this SQL query is correctly answering the original question.
        
        Original question: "{original_query}"
        
        SQL query:
        ```sql
        {sql_query}
        ```
        
        Sample results (first 10 rows):
        {results_sample}
        
        Total rows returned: {len(sample_results)}
        
        Please analyze:
        1. Do the results contain the information needed to answer the original question?
        2. Are there any obvious issues with the query based on the results?
        3. Does the data format match what would be expected for the question?
        4. Are there any missing columns or unexpected NULL values?
        
        Format your response as a JSON object with these keys:
        - is_correct: boolean indicating whether the query correctly answers the question
        - issues: list of identified issues (empty if none)
        - suggestions: list of suggestions to improve the query
        - confidence: score from 0-1 indicating confidence in the query correctness
        """
        
        return self.run_with_json_output(prompt, temperature=0.3)


class AgentPerformanceMonitor:
    """Monitors and improves agent performance over time"""
    
    def __init__(self, storage_path="agent_performance.json"):
        self.storage_path = storage_path
        self.performance_data = self._load_data()
        
    def _load_data(self):
        """Load existing performance data if available"""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except:
            return {
                "schema_analysis": {"success": 0, "failure": 0, "avg_time": 0},
                "sql_generation": {"success": 0, "failure": 0, "avg_time": 0},
                "sql_validation": {"success": 0, "failure": 0, "avg_time": 0},
                "query_testing": {"success": 0, "failure": 0, "avg_time": 0},
                "data_analysis": {"success": 0, "failure": 0, "avg_time": 0},
                "queries": {}
            }
            
    def log_query(self, query, agent_results, execution_times, was_successful):
        """Log performance data for a query"""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Store query performance
        if query_hash not in self.performance_data["queries"]:
            self.performance_data["queries"][query_hash] = {
                "text": query,
                "attempts": 0,
                "success": 0,
                "failure": 0,
                "avg_time": 0,
                "last_sql": None
            }
        
        # Update stats
        query_data = self.performance_data["queries"][query_hash]
        query_data["attempts"] += 1
        if was_successful:
            query_data["success"] += 1
        else:
            query_data["failure"] += 1
        
        # Update agent-specific stats
        for agent, time_taken in execution_times.items():
            if agent in self.performance_data:
                agent_data = self.performance_data[agent]
                if was_successful:
                    agent_data["success"] += 1
                else:
                    agent_data["failure"] += 1
                
                # Update average time
                current_avg = agent_data["avg_time"]
                total_runs = agent_data["success"] + agent_data["failure"]
                agent_data["avg_time"] = (current_avg * (total_runs - 1) + time_taken) / total_runs
        
        # Save the latest SQL
        if "sql_query" in agent_results:
            query_data["last_sql"] = agent_results["sql_query"]
            
        # Save updated data
        self._save_data()
        
    def _save_data(self):
        """Save performance data to disk"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.performance_data, f)
            
    def get_recommendations(self):
        """Analyze performance data and provide recommendations"""
        recommendations = []
        
        # Identify problematic queries (high failure rate)
        problem_queries = []
        for query_hash, data in self.performance_data["queries"].items():
            if data["attempts"] >= 3 and data["failure"] / data["attempts"] > 0.5:
                problem_queries.append((data["text"], data["failure"] / data["attempts"]))
        
        if problem_queries:
            recommendations.append({
                "type": "problem_queries",
                "description": "These queries have high failure rates and may need templates",
                "queries": problem_queries
            })
            
        # Check agent performance
        for agent, data in {k: v for k, v in self.performance_data.items() if k != "queries"}.items():
            total = data["success"] + data["failure"]
            if total > 0 and data["failure"] / total > 0.3:
                recommendations.append({
                    "type": "agent_issue",
                    "agent": agent,
                    "description": f"{agent} has a high failure rate ({data['failure']}/{total})",
                    "suggestion": "Consider refining the system prompt or adding more examples"
                })
                
        return recommendations


def validate_and_fix_sql(sql_query, tables_info):
    """
    Validates and fixes common SQL errors in LLM-generated queries.
    
    Args:
        sql_query: The SQL query to validate and fix
        tables_info: Dictionary with table schema information
        
    Returns:
        Tuple of (fixed_query, list_of_changes, is_safe)
    """
    changes = []
    is_safe = True
    fixed_query = sql_query
    
    # Check for basic SQL injection patterns
    unsafe_patterns = [
        ";", "--", "DROP", "TRUNCATE", "DELETE FROM", "ALTER TABLE", 
        "CREATE USER", "GRANT ALL"
    ]
    for pattern in unsafe_patterns:
        if pattern.lower() in sql_query.lower():
            # If it's part of a legitimate SQL operation with proper context, it might be fine
            # Otherwise flag it as unsafe
            context_check = check_safe_context(pattern, sql_query)
            if not context_check:
                is_safe = False
                changes.append(f"Potentially unsafe SQL pattern detected: {pattern}")
                
    # Check for basic SQL syntax issues
    syntax_checks = [
        # Missing FROM clause
        {"pattern": r"SELECT.*(?<!FROM)", "fix": None, "message": "SELECT without FROM clause"},
        # Missing comma between columns
        {"pattern": r'[a-zA-Z0-9_]+"?\s+"?[a-zA-Z0-9_]', "fix": None, "message": "Possible missing comma between columns"},
        # Mismatched parentheses
        {"pattern": None, "fix": None, "message": "Mismatched parentheses", 
         "check": lambda q: q.count('(') != q.count(')')},
        # GROUP BY inconsistencies
        {"pattern": r"SELECT.*,.*\bGROUP\s+BY\b", "fix": None, "message": "Possible GROUP BY inconsistency with non-aggregated columns"},
        # Ambiguous column references in joins
        {"pattern": r"JOIN.*SELECT", "fix": None, "message": "Possible ambiguous column references in JOIN"}
    ]
    
    import re
    for check in syntax_checks:
        if check.get("pattern") and re.search(check["pattern"], sql_query, re.IGNORECASE | re.DOTALL):
            changes.append(f"SQL syntax issue detected: {check['message']}")
        elif check.get("check") and check["check"](sql_query):
            changes.append(f"SQL syntax issue detected: {check['message']}")
    
    # Fix common column reference errors
    available_columns = {}
    all_columns = []  # For semantic matching, keep a list of all columns
    table_column_map = {}  # Map tables to their columns for better context-aware matching
    
    for table, schema in tables_info.items():
        cols = schema['column_name'].tolist()
        available_columns[table] = [col.lower() for col in cols]
        # Add all columns to a flat list for semantic matching
        all_columns.extend(cols)
        table_column_map[table] = cols
    
    # Define common payroll/employee column mappings (domain-specific knowledge)
    common_column_mappings = {
        "net_monthly_pay": ["monthly_salary", "net_salary", "net_pay", "monthly_pay", "salary_net", "salary"],
        "gross_monthly_pay": ["gross_salary", "gross_pay", "monthly_gross", "salary_gross", "total_salary"],
        "salary": ["monthly_salary", "net_salary", "gross_salary", "net_pay", "gross_pay", "pay", "wage", "earnings"],
        "employee_name": ["name", "full_name", "emp_name", "staff_name", "personnel_name"],
        "employee_id": ["emp_id", "id", "staff_id", "personnel_id", "worker_id"],
        "department": ["dept", "division", "team", "unit", "section"],
        "position": ["job_title", "role", "title", "designation", "job_role"],
        "hire_date": ["join_date", "joining_date", "employment_date", "start_date"],
    }
    
    # Extract column references from the SQL query
    # This is a simplified approach; in practice, you'd use a SQL parser
    try:
        # Remove string literals to avoid matching text inside quotes
        no_strings_query = re.sub(r"'[^']*'", "", sql_query)
        no_strings_query = re.sub(r'"[^"]*"', "", no_strings_query)
        
        # Find column references 
        # Look for patterns like: table.column, "column", [column], `column`
        column_refs = re.findall(r'(?:[\w]+\.)?[\[\"\`]?(\w+)[\]\"\`]?', no_strings_query)
        
        # First pass: Check referenced tables and columns for exact case-insensitive matches
        for table in tables_info.keys():
            if table.lower() in sql_query.lower():
                # The table is used in the query, check its columns
                for col in available_columns[table]:
                    # Look for mismatched column names (case sensitivity issues)
                    if col not in sql_query and col.lower() in sql_query.lower():
                        incorrect_col = re.search(r'\b' + re.escape(col.lower()) + r'\b', sql_query.lower())
                        if incorrect_col:
                            # Get the actual (incorrectly cased) text from the original query
                            span = incorrect_col.span()
                            actual_text = sql_query[span[0]:span[1]]
                            fixed_query = fixed_query.replace(actual_text, col)
                            changes.append(f"Fixed column name case: '{actual_text}' to '{col}'")
        
        # Second pass: Look for common payroll column name mismatches
        for actual_col, synonyms in common_column_mappings.items():
            for synonym in synonyms:
                # Check if the synonym is in the query but not as part of another word
                pattern = r'\b' + re.escape(synonym) + r'\b'
                match = re.search(pattern, fixed_query, re.IGNORECASE)
                if match:
                    # Check if the actual column exists in any table
                    for table, columns in table_column_map.items():
                        if actual_col in columns:
                            # Found a match - replace the synonym with the actual column
                            span = match.span()
                            synonym_text = fixed_query[span[0]:span[1]]
                            
                            # Replace with proper quoting based on context
                            if '"' in fixed_query:
                                # If we have double quotes in the query, maintain that style
                                # Check if the term is already quoted
                                if fixed_query[span[0]-1] == '"' and fixed_query[span[1]] == '"':
                                    replacement = f'"{actual_col}"' 
                                    fixed_query = fixed_query[:span[0]-1] + replacement + fixed_query[span[1]+1:]
                                else:
                                    replacement = f'"{actual_col}"'
                                    fixed_query = fixed_query[:span[0]] + replacement + fixed_query[span[1]:]
                            else:
                                # Simple replacement
                                fixed_query = fixed_query[:span[0]] + actual_col + fixed_query[span[1]:]
                            
                            changes.append(f"Fixed column name using domain knowledge: '{synonym_text}' to '{actual_col}'")
                            break
        
        # Third pass: Advanced semantic matching for other column references  
        for ref in column_refs:
            if len(ref) < 3:  # Skip very short references
                continue
                
            # Check if this reference exactly matches any known column (case-insensitive)
            exact_match = False
            for col in all_columns:
                if ref.lower() == col.lower():
                    exact_match = True
                    break
                    
            if not exact_match:
                # Try semantic matching - using multiple similarity metrics
                best_match = None
                best_score = 0.55  # Lower threshold to catch more matches
                
                for col in all_columns:
                    # Simple character overlap similarity
                    char_similarity = len(set(ref.lower()) & set(col.lower())) / max(len(ref), len(col))
                    
                    # Check for substring matches
                    substring_bonus = 0
                    if ref.lower() in col.lower() or col.lower() in ref.lower():
                        substring_bonus = 0.2
                    
                    # Check for plural forms and common suffixes
                    morphology_bonus = 0
                    if ref.lower() + 's' == col.lower() or ref.lower() == col.lower() + 's':
                        morphology_bonus = 0.2
                    elif ref.lower() + 'es' == col.lower() or ref.lower() == col.lower() + 'es':
                        morphology_bonus = 0.15
                    elif ref.lower() + '_id' == col.lower() or ref.lower() == col.lower() + '_id':
                        morphology_bonus = 0.15
                    
                    # Check for word boundary matches (for multi-word columns)
                    word_boundary_bonus = 0
                    ref_words = ref.lower().split('_')
                    col_words = col.lower().split('_')
                    common_words = set(ref_words) & set(col_words)
                    if common_words:
                        word_boundary_bonus = 0.1 * len(common_words) / max(len(ref_words), len(col_words))
                    
                    # Calculate total similarity
                    similarity = char_similarity + substring_bonus + morphology_bonus + word_boundary_bonus
                    
                    if similarity > best_score:
                        best_match = col
                        best_score = similarity
                
                if best_match:
                    # Try to replace the reference with the best match
                    pattern = r'\b' + re.escape(ref) + r'\b'
                    new_query = re.sub(pattern, best_match, fixed_query)
                    if new_query != fixed_query:
                        changes.append(f"Fixed column name using semantic matching: '{ref}' to '{best_match}'")
                        fixed_query = new_query
    except Exception as e:
        changes.append(f"Error in semantic column matching: {str(e)}")
    
    return fixed_query, changes, is_safe


def check_safe_context(pattern, sql_query):
    """
    Check if a potentially unsafe pattern appears in a safe context.
    
    Args:
        pattern: The pattern to check
        sql_query: The SQL query
        
    Returns:
        Boolean indicating if the pattern is in a safe context
    """
    # Add context-specific checks for each pattern
    if pattern == ";":
        # Semi-colons at the end of a query are fine
        if sql_query.strip().endswith(";"):
            return True
    elif pattern == "--":
        # Comments might be legitimate
        return True
    elif pattern == "DROP" and "DROP TABLE" in sql_query:
        # DROP TABLE is unsafe
        return False
    
    # Default to not safe if we don't have a specific rule
    return False


def simplified_agent_workflow(question, tables_info, conn, db_type="postgresql", model_provider="openai"):
    """
    Execute a simplified agent workflow that uses a combined agent for schema analysis and SQL generation.
    
    Args:
        question: User's natural language question
        tables_info: Dictionary with table schema information
        conn: Database connection
        db_type: Database type (postgresql or mysql)
        model_provider: Model provider to use (openai, anthropic, mistral)
        
    Returns:
        Dictionary with results from all agents and final analysis
    """
    # Initialize agents with the selected provider
    combined_agent = CombinedSQLAgent(provider=model_provider)
    validation_agent = SQLValidationAgent(provider=model_provider)
    analysis_agent = DataAnalysisAgent(provider=model_provider)
    
    # Step 1: Generate SQL query directly from question and schema in one step
    combined_result = combined_agent.generate_sql_query(question, tables_info, db_type)
    sql_query = combined_result.get('sql_query', '')
    column_mappings = combined_result.get('column_mappings', {})
    
    # Step 2: Validate and fix SQL query
    validation_result = validation_agent.validate_and_fix_sql(question, sql_query, tables_info, db_type)
    
    # Use the fixed SQL if available and valid
    if validation_result.get('is_valid', False):
        sql_query = validation_result.get('fixed_sql', sql_query)
    
    # Step 3: Execute SQL query
    if not sql_query:
        return {
            "error": "Failed to generate SQL query",
            "combined_result": combined_result,
            "validation_result": validation_result
        }
    
    try:
        results = database.execute_sql_query(conn, sql_query)
    except Exception as e:
        # If SQL execution fails, apply manual semantic matching fix using our standalone function
        try:
            fixed_query, changes, is_safe = validate_and_fix_sql(sql_query, tables_info)
            if changes and is_safe:
                # Try with the fixed query
                results = database.execute_sql_query(conn, fixed_query)
                sql_query = fixed_query
            else:
                # If no changes were made or query is unsafe, report the original error
                return {
                    "error": f"Error executing SQL query: {str(e)}",
                    "combined_result": combined_result,
                    "validation_result": validation_result,
                    "sql_query": sql_query
                }
        except Exception as e2:
            # If both attempts fail, report the original error
            return {
                "error": f"Error executing SQL query: {str(e)}",
                "combined_result": combined_result,
                "validation_result": validation_result,
                "sql_query": sql_query
            }
    
    # Step 4: Analyze data and provide insights
    data_analysis = analysis_agent.analyze_data(question, sql_query, results)
    explanation = analysis_agent.explain_analysis(question, sql_query, results)
    
    # Compile all results
    return {
        "combined_result": combined_result,
        "validation_result": validation_result,
        "sql_query": sql_query,
        "data": results,
        "analysis": {
            **data_analysis,
            "explanation": explanation
        }
    }

def multi_agent_workflow(question, tables_info, conn, db_type="postgresql", model_provider="openai", model=None):
    """
    Execute a hybrid workflow that uses standard SQL generation and agent-based analysis.
    
    Args:
        question: User's natural language question
        tables_info: Dictionary with table schema information
        conn: Database connection
        db_type: Database type (postgresql or mysql)
        model_provider: Model provider to use (openai, anthropic, mistral, ollama)
        model: Specific model to use (if None, will use provider default)
        
    Returns:
        Dictionary with results from all agents and final analysis
    """
    execution_times = {}
    
    # Step 1: Use standard NLP to SQL approach for query generation
    import time
    import database  # Import here to avoid circular imports
    import nlp  # Import here to avoid circular imports
    
    # Generate SQL using the standard robust approach
    start_time = time.time()
    sql_query = nlp.natural_language_to_sql(question, tables_info, db_type=db_type)
    execution_times["sql_generation"] = time.time() - start_time
    
    # Create a validation agent to check the query
    validation_agent = SQLValidationAgent(provider=model_provider, model=model)
    start_time = time.time()
    validation_result = validation_agent.validate_and_fix_sql(question, sql_query, tables_info, db_type)
    execution_times["sql_validation"] = time.time() - start_time
    
    # Use the fixed SQL if available and valid
    if validation_result.get('is_valid', False):
        sql_query = validation_result.get('fixed_sql', sql_query)
    
    # Step 2: Execute SQL query
    if not sql_query:
        return {
            "error": "Failed to generate SQL query",
            "validation_result": validation_result
        }
    
    try:
        start_time = time.time()
        results = database.execute_sql_query(conn, sql_query)
        execution_times["query_execution"] = time.time() - start_time
    except Exception as e:
        # If SQL execution fails, apply manual semantic matching fix using our standalone function
        try:
            fixed_query, changes, is_safe = validate_and_fix_sql(sql_query, tables_info)
            if changes and is_safe:
                # Try with the fixed query
                start_time = time.time()
                results = database.execute_sql_query(conn, fixed_query)
                execution_times["query_execution"] = time.time() - start_time
                sql_query = fixed_query
                
                # Create a new validation_result that includes our auto-fixes
                validation_result = {
                    "is_valid": True,
                    "fixed_sql": fixed_query,
                    "auto_fixes": [f"Auto-fix: {change}" for change in changes],
                    "is_safe": is_safe
                }
            else:
                # If no changes were made or query is unsafe, report the original error
                return {
                    "error": f"Error executing SQL query: {str(e)}",
                    "validation_result": validation_result,
                    "sql_query": sql_query
                }
        except Exception as e2:
            # If both attempts fail, report the original error
            return {
                "error": f"Error executing SQL query: {str(e)}\nSecondary error: {str(e2)}",
                "validation_result": validation_result,
                "sql_query": sql_query
            }
    
    # Step 3: Use specialized agents for the remaining steps (testing and analysis)
    testing_agent = QueryTestingAgent(provider=model_provider, model=model)
    analysis_agent = DataAnalysisAgent(provider=model_provider, model=model)
    
    # Test query results
    start_time = time.time()
    test_result = testing_agent.test_query(question, sql_query, results)
    execution_times["query_testing"] = time.time() - start_time
    
    # Analyze data and provide insights
    start_time = time.time()
    data_analysis = analysis_agent.analyze_data(question, sql_query, results)
    explanation = analysis_agent.explain_analysis(question, sql_query, results)
    execution_times["data_analysis"] = time.time() - start_time
    
    # Store streaming explanation method for use in UI
    explanation_streaming = lambda streamlit_placeholder: analysis_agent.explain_analysis_streaming(
        question, sql_query, results, None, streamlit_placeholder)
    
    # Extract metadata about the query using NLP module
    query_metadata = nlp.extract_query_metadata(question, sql_query, tables_info)
    
    # Compile all results
    return {
        "validation_result": validation_result,
        "test_result": test_result,
        "sql_query": sql_query,
        "data": results,
        "analysis": {
            **data_analysis,
            "explanation": explanation,
            "explanation_streaming": explanation_streaming
        },
        "query_metadata": query_metadata,
        "execution_times": execution_times
    }

def enhanced_multi_agent_workflow(question, tables_info, conn, db_type="postgresql", 
                           use_specialized_sql_model=False, specialized_sql_model="gpt-4o",
                           model_provider="openai", model=None):
    """
    Execute an enhanced multi-agent workflow that uses standard SQL generation with specialized model
    and agent-based analysis.
    
    Args:
        question: User's natural language question
        tables_info: Dictionary with table schema information
        conn: Database connection
        db_type: Database type (postgresql or mysql)
        use_specialized_sql_model: Whether to use a specialized model for SQL generation
        specialized_sql_model: Name of the model to use for SQL generation 
        model_provider: Model provider to use (openai, anthropic, mistral, ollama)
        model: Specific model to use (if None, will use provider default)
        
    Returns:
        Dictionary with results from all agents and final analysis
    """
    execution_times = {}
    
    # Step 1: Use standard NLP to SQL approach for query generation with specialized model if configured
    import time
    import database  # Import here to avoid circular imports
    import nlp  # Import here to avoid circular imports
    
    # If use_specialized_sql_model is True, temporarily override the model in the NLP module
    current_model = None
    if use_specialized_sql_model:
        # Save current model if needed for restoring later
        if hasattr(nlp, 'MODEL'):
            current_model = nlp.MODEL
            
        # Temporarily set the model to the specialized one - for OpenAI only
        # Other providers use model_clients.py directly
        if model_provider == "openai":
            nlp.MODEL = specialized_sql_model
    
    # Generate SQL using the standard robust approach
    start_time = time.time()
    # If using a provider other than OpenAI with specialized model, we need to handle it differently
    if use_specialized_sql_model and model_provider != "openai":
        # Import model_clients to use the appropriate provider's client
        import model_clients
        
        # Get the right client based on provider and model
        client = model_clients.get_model_client(model_provider, specialized_sql_model)
        
        # Prepare the schema information for the prompt
        schema_text = ""
        for table_name, schema in tables_info.items():
            schema_text += f"Table: {table_name}\nColumns:\n"
            # Handle different schema formats
            if "column_name" in schema.columns:
                for _, row in schema.iterrows():
                    schema_text += f"- {row['column_name']} ({row['data_type']})\n"
            elif "Column Name" in schema.columns:
                for _, row in schema.iterrows():
                    schema_text += f"- {row['Column Name']} ({row['Data Type']})\n"
            else:
                # Use whatever columns are available
                columns = schema.columns.tolist()
                for _, row in schema.iterrows():
                    schema_text += f"- {row[columns[0]]} ({row[columns[1] if len(columns) > 1 else 'unknown']})\n"
            schema_text += "\n"
        
        # Create the prompt
        messages = [
            {"role": "system", "content": f"You are an expert SQL developer. Convert natural language questions to valid {db_type.upper()} SQL queries. Only return valid SQL, no explanations."},
            {"role": "user", "content": f"Database schema:\n{schema_text}\n\nGenerate a valid {db_type.upper()} SQL query for this question: {question}"}
        ]
        
        # Generate the SQL using the specialized model
        response = client.generate_completion(messages, temperature=0.1)
        
        # Extract the SQL from the response
        sql_query = response.strip()
        # Remove any markdown formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
    else:
        # Use the standard NLP approach
        sql_query = nlp.natural_language_to_sql(question, tables_info, db_type=db_type)
    
    execution_times["sql_generation"] = time.time() - start_time
    
    # Restore original model if we changed it
    if use_specialized_sql_model and current_model and model_provider == "openai":
        nlp.MODEL = current_model
    
    # Create a validation agent to check the query (with specialized model if configured)
    validation_agent = SQLValidationAgent(
        provider=model_provider,
        model=specialized_sql_model if use_specialized_sql_model else model
    )
    
    start_time = time.time()
    validation_result = validation_agent.validate_and_fix_sql(question, sql_query, tables_info, db_type)
    execution_times["sql_validation"] = time.time() - start_time
    
    # Use the fixed SQL if available and valid
    if validation_result.get('is_valid', False):
        sql_query = validation_result.get('fixed_sql', sql_query)
    
    # Step 2: Execute SQL query
    if not sql_query:
        return {
            "error": "Failed to generate SQL query",
            "validation_result": validation_result
        }
    
    try:
        start_time = time.time()
        results = database.execute_sql_query(conn, sql_query)
        execution_times["query_execution"] = time.time() - start_time
    except Exception as e:
        # If SQL execution fails, apply manual semantic matching fix using our standalone function
        try:
            fixed_query, changes, is_safe = validate_and_fix_sql(sql_query, tables_info)
            if changes and is_safe:
                # Try with the fixed query
                start_time = time.time()
                results = database.execute_sql_query(conn, fixed_query)
                execution_times["query_execution"] = time.time() - start_time
                sql_query = fixed_query
                
                # Create a new validation_result that includes our auto-fixes
                validation_result = {
                    "is_valid": True,
                    "fixed_sql": fixed_query,
                    "auto_fixes": [f"Auto-fix: {change}" for change in changes],
                    "is_safe": is_safe
                }
            else:
                # If no changes were made or query is unsafe, report the original error
                return {
                    "error": f"Error executing SQL query: {str(e)}",
                    "validation_result": validation_result,
                    "sql_query": sql_query
                }
        except Exception as e2:
            # If both attempts fail, report the original error
            return {
                "error": f"Error executing SQL query: {str(e)}\nSecondary error: {str(e2)}",
                "validation_result": validation_result,
                "sql_query": sql_query
            }
    
    # Step 3: Use specialized agents for the remaining steps (testing and analysis)
    # Use the specialized model for these agents if it's enabled, otherwise use the general model
    model_name = specialized_sql_model if use_specialized_sql_model else model
    testing_agent = QueryTestingAgent(provider=model_provider, model=model_name)
    analysis_agent = DataAnalysisAgent(provider=model_provider, model=model_name)
    
    # Test query results
    start_time = time.time()
    test_result = testing_agent.test_query(question, sql_query, results)
    execution_times["query_testing"] = time.time() - start_time
    
    # Analyze data and provide insights
    start_time = time.time()
    data_analysis = analysis_agent.analyze_data(question, sql_query, results)
    explanation = analysis_agent.explain_analysis(question, sql_query, results)
    execution_times["data_analysis"] = time.time() - start_time
    
    # Store streaming explanation method for use in UI
    explanation_streaming = lambda streamlit_placeholder: analysis_agent.explain_analysis_streaming(
        question, sql_query, results, None, streamlit_placeholder)
    
    # Extract metadata about the query using NLP module
    query_metadata = nlp.extract_query_metadata(question, sql_query, tables_info)
    
    # Compile all results
    return {
        "validation_result": validation_result,
        "test_result": test_result,
        "sql_query": sql_query,
        "data": results,
        "analysis": {
            **data_analysis,
            "explanation": explanation,
            "explanation_streaming": explanation_streaming
        },
        "query_metadata": query_metadata,
        "execution_times": execution_times
    }