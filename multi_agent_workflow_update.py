from agents import SQLValidationAgent, QueryTestingAgent, DataAnalysisAgent, validate_and_fix_sql

def multi_agent_workflow(question, tables_info, conn, db_type="postgresql", model_provider="openai"):
    """
    Execute a hybrid workflow that uses standard SQL generation and agent-based analysis.
    
    Args:
        question: User's natural language question
        tables_info: Dictionary with table schema information
        conn: Database connection
        db_type: Database type (postgresql or mysql)
        model_provider: Model provider to use (openai, anthropic, mistral)
        
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
    validation_agent = SQLValidationAgent(provider=model_provider)
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
    testing_agent = QueryTestingAgent(provider=model_provider)
    analysis_agent = DataAnalysisAgent(provider=model_provider)
    
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