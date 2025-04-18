import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
import database
import nlp
import visualization
import utils
import reports
import agents
import web_scraper
import image_analyzer

# Page configuration
st.set_page_config(
    page_title="QuickQuery AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'conn' not in st.session_state:
    st.session_state.conn = None
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'tables_info' not in st.session_state:
    st.session_state.tables_info = {}
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
# PDF source viewer state
if 'show_pdf_source' not in st.session_state:
    st.session_state.show_pdf_source = False
if 'selected_tables' not in st.session_state:
    st.session_state.selected_tables = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'current_sql' not in st.session_state:
    st.session_state.current_sql = ""
if 'current_results' not in st.session_state:
    st.session_state.current_results = pd.DataFrame()
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = True
if 'viz_debug_info' not in st.session_state:
    st.session_state.viz_debug_info = {}
    
# Initialize reports-related session state
if 'data_source_mode' not in st.session_state:
    st.session_state.data_source_mode = "tables"  # Either "tables", "reports", "web", or "pdf"

# Initialize PDF-related session state
if 'pdf_collections' not in st.session_state:
    st.session_state.pdf_collections = {}  # Store PDF collection metadata
    
if 'selected_pdf_collections' not in st.session_state:
    st.session_state.selected_pdf_collections = []  # Store selected PDF collections
# Initialize custom reports
reports.initialize_reports()
if 'available_reports' not in st.session_state:
    st.session_state.available_reports = reports.get_report_list()
if 'selected_reports' not in st.session_state:
    st.session_state.selected_reports = []
if 'reports_info' not in st.session_state:
    st.session_state.reports_info = {}
if 'report_view_name' not in st.session_state:
    st.session_state.report_view_name = None
# Web scraping collections
if 'web_collections' not in st.session_state:
    st.session_state.web_collections = web_scraper.get_all_collections() if 'web_scraper' in globals() else []
if 'selected_collection' not in st.session_state:
    st.session_state.selected_collection = None
if 'selected_collections' not in st.session_state:
    st.session_state.selected_collections = []
if 'collection_stats' not in st.session_state:
    st.session_state.collection_stats = {}
if 'web_results' not in st.session_state:
    st.session_state.web_results = pd.DataFrame()

# Report-related state
if 'selected_tables_for_report' not in st.session_state:
    st.session_state.selected_tables_for_report = []
if 'reset_report_form' not in st.session_state:
    st.session_state.reset_report_form = False
if 'use_multi_agent' not in st.session_state:
    st.session_state.use_multi_agent = False
if 'use_enhanced_multi_agent' not in st.session_state:
    st.session_state.use_enhanced_multi_agent = False
if 'use_specialized_sql_model' not in st.session_state:
    st.session_state.use_specialized_sql_model = False
if 'specialized_sql_model' not in st.session_state:
    st.session_state.specialized_sql_model = "gpt-4o"
if 'model_provider' not in st.session_state:
    st.session_state.model_provider = "openai"

# Handle reset_report_form flag
if st.session_state.reset_report_form:
    # Delete the keys to remove the widgets
    for key in ['new_report_name', 'new_report_desc', 'new_report_query']:
        if key in st.session_state:
            del st.session_state[key]
    # Reset the flag
    st.session_state.reset_report_form = False

# Title and icon
st.title("🔍 QuickQuery AI")

# Sidebar
with st.sidebar:
    # Debug mode toggle in sidebar
    with st.expander("Advanced Options", expanded=False):
        st.session_state['debug_mode'] = st.checkbox("Enable Debug Mode", value=st.session_state.get('debug_mode', True))
        if st.session_state['debug_mode']:
            st.info("Debug mode is enabled. You will see detailed error messages and debug information.")
        
        # Initialize viz_debug_info if it doesn't exist
        if 'viz_debug_info' not in st.session_state:
            st.session_state['viz_debug_info'] = {}
            
        st.markdown("---")
        
        # Data source selection at the top level
    st.subheader("Data Source")
    top_level_data_source = st.radio(
        "Choose data source type",
        ["Database", "Website Chat", "PDF Documents", "Image Analysis"],
        index=0 if st.session_state.data_source_mode in ["tables", "reports"] 
              else (1 if st.session_state.data_source_mode == "web" 
                   else (2 if st.session_state.data_source_mode == "pdf" else 3)),
        help="Select which type of data source to interact with"
    )
    
    # Database sub-selection (moved to main area, we just set defaults here)
    if top_level_data_source == "Database":
        # Map radio options to mode values
        mode_mapping = {
            "Database Tables": "tables",
            "Custom Reports": "reports"
        }
        
        # Reverse mapping for session state values to radio options
        reverse_mapping = {
            "tables": "Database Tables",
            "reports": "Custom Reports"
        }
        
        # Set default mode if not connected
        if not hasattr(st.session_state, 'conn') or not st.session_state.conn:
            new_mode = "tables"
        else:
            # This will be overridden by the radio selection in the main area
            new_mode = st.session_state.data_source_mode
    elif top_level_data_source == "Website Chat":
        new_mode = "web"
    elif top_level_data_source == "PDF Documents":
        new_mode = "pdf"
    else:  # Image Analysis
        new_mode = "image"
    
    # Update data source mode in session state and reset results when switching modes
    previous_mode = st.session_state.data_source_mode
    
    # Check if mode has changed and reset results if needed
    if previous_mode != new_mode:
        st.session_state.current_results = pd.DataFrame()
        st.session_state.current_query = ""
        st.session_state.current_sql = ""
        st.session_state.web_results = pd.DataFrame()
        
        # Additional resets based on the specific mode
        if new_mode == "web":
            # Reset database and PDF-related selections when switching to web mode
            st.session_state.selected_tables = []
            st.session_state.selected_reports = []
            st.session_state.selected_pdf_collections = []
        elif new_mode == "tables":
            # Reset web, reports, and PDF related selections when switching to tables mode
            st.session_state.selected_collection = None
            st.session_state.selected_reports = []
            st.session_state.selected_pdf_collections = []
        elif new_mode == "reports":
            # Reset web, tables, and PDF related selections when switching to reports mode
            st.session_state.selected_collection = None
            st.session_state.selected_tables = []
            st.session_state.selected_pdf_collections = []
        elif new_mode == "pdf":
            # Reset web, tables, and reports related selections when switching to PDF mode
            st.session_state.selected_collection = None
            st.session_state.selected_tables = []
            st.session_state.selected_reports = []
        elif new_mode == "image":
            # Reset web, tables, reports, and PDF related selections when switching to Image Analysis mode
            st.session_state.selected_collection = None
            st.session_state.selected_tables = []
            st.session_state.selected_reports = []
            st.session_state.selected_pdf_collections = []
            
    # Update the session state with the new mode
    st.session_state.data_source_mode = new_mode
    
    st.markdown("---")
    
    # Add Query Processing Settings in a collapsible expander
    with st.expander("Query Processing Settings", expanded=False):
        st.caption("Configure AI models and processing modes for query handling")
        
        # Agent processing options
        agent_mode = st.radio(
            "Agent Processing Mode",
            ["Standard", "Multi-Agent", "Enhanced Multi-Agent"],
            index=0 if not st.session_state.use_multi_agent else (2 if st.session_state.use_enhanced_multi_agent else 1),
            help="Choose the processing mode for natural language queries"
        )
        
        # Set session state based on selection
        st.session_state.use_multi_agent = agent_mode != "Standard"
        st.session_state.use_enhanced_multi_agent = agent_mode == "Enhanced Multi-Agent"
        
        # Model provider selection
        model_providers = ["openai", "anthropic", "mistral", "ollama", "custom"]
        st.session_state.model_provider = st.selectbox(
            "Model Provider",
            model_providers,
            index=model_providers.index(st.session_state.get('model_provider', "openai")),
            help="Select the AI model provider to use for all agents"
        )
        
        # Get available models for the selected provider
        from model_clients import list_available_models
        
        # Show model selection for the selected provider
        available_models = list_available_models(st.session_state.model_provider).get(st.session_state.model_provider, [])
        
        # Set default model based on provider
        default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "mistral": "mistral-large-latest",
            "ollama": "llama3",
            "custom": "custom-model"
        }
        
        # Get the current default model
        default_model = st.session_state.get(f"{st.session_state.model_provider}_model", 
                                            default_models.get(st.session_state.model_provider))
        
        # Set default index
        try:
            default_index = available_models.index(default_model)
        except (ValueError, IndexError):
            default_index = 0
            
        # Add model selection dropdown
        if available_models:
            selected_model = st.selectbox(
                f"{st.session_state.model_provider.title()} Model",
                available_models,
                index=default_index,
                help=f"Select which {st.session_state.model_provider.title()} model to use for generating responses"
            )
            
            # Store the selected model in session state with a provider-specific key
            st.session_state[f"{st.session_state.model_provider}_model"] = selected_model
            
            # Also store in a generic current_model field for easy access
            st.session_state.current_model = selected_model
        
        # Display specialized SQL model selection when using enhanced multi-agent
        if st.session_state.use_enhanced_multi_agent:
            st.session_state.use_specialized_sql_model = st.checkbox(
                "Use Specialized SQL LLM", 
                value=st.session_state.get('use_specialized_sql_model', False),
                help="Enable specialized model for SQL generation"
            )
            
            # Show SQL model selection if specialized model is enabled
            if st.session_state.use_specialized_sql_model:
                # Show models based on selected provider
                if st.session_state.model_provider == "openai":
                    sql_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
                elif st.session_state.model_provider == "anthropic":
                    sql_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
                elif st.session_state.model_provider == "mistral":
                    sql_models = ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]
                else:  # ollama
                    # Get available Ollama models
                    try:
                        from model_clients import list_available_models
                        ollama_models = list_available_models("ollama").get("ollama", ["llama3"])
                        sql_models = ollama_models
                    except:
                        # Default Ollama models if we can't get the list
                        sql_models = ["llama3", "codellama", "mistral"]
                
                # Default index based on provider
                default_model = st.session_state.get('specialized_sql_model', "gpt-4o")
                try:
                    index = sql_models.index(default_model)
                except ValueError:
                    index = 0
                
                st.session_state.specialized_sql_model = st.selectbox(
                    "SQL Generation Model",
                    sql_models,
                    index=index,
                    help="Select specialized model for SQL generation"
                )
                
        st.caption(f"""
        ℹ️ **Agent Processing Info**:
        - **Multi-Agent**: Uses specialized agents to analyze schema, generate SQL, and analyze data
        - **Enhanced Multi-Agent**: Adds SQL validation and testing agents for improved reliability
        - **Model Provider**: {st.session_state.model_provider.capitalize()}
        """)
    
    st.markdown("---")
    
    # Add Model Settings expander
    with st.expander("Model API Settings", expanded=False):
        st.caption("Configure API keys for different model providers")
        
        # Load environment variables from .env file
        utils.load_env_vars()
        
        # OpenAI settings
        st.subheader("OpenAI Settings")
        # Get existing key if available, otherwise empty string
        current_openai_key = utils.get_api_key("openai") or ""
        openai_key_masked = "********" if current_openai_key else ""
        
        # Display if key is already set
        if current_openai_key:
            st.success("OpenAI API key is set")
        
        # OpenAI API Key input
        openai_key = st.text_input(
            "OpenAI API Key", 
            value="",
            type="password",
            help="Enter your OpenAI API Key (will be saved to .env file)",
            placeholder="Enter API key" if not current_openai_key else "Key already saved"
        )
        
        # Save/Delete buttons for OpenAI
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save OpenAI Key", key="save_openai"):
                if openai_key:
                    if utils.save_api_key("openai", openai_key):
                        st.session_state['openai_api_key'] = openai_key
                        st.success("OpenAI API key saved to .env file")
                        st.rerun()
                    else:
                        st.error("Failed to save OpenAI API key")
                else:
                    st.warning("Please enter an API key to save")
        
        with col2:
            if st.button("Delete OpenAI Key", key="delete_openai"):
                if utils.delete_api_key("openai"):
                    if 'openai_api_key' in st.session_state:
                        del st.session_state['openai_api_key']
                    st.success("OpenAI API key deleted")
                    st.rerun()
                else:
                    st.warning("No OpenAI API key to delete")
        
        st.markdown("---")
        
        # Anthropic settings
        st.subheader("Anthropic Settings")
        # Get existing key if available, otherwise empty string
        current_anthropic_key = utils.get_api_key("anthropic") or ""
        anthropic_key_masked = "********" if current_anthropic_key else ""
        
        # Display if key is already set
        if current_anthropic_key:
            st.success("Anthropic API key is set")
        
        # Anthropic API Key input
        anthropic_key = st.text_input(
            "Anthropic API Key", 
            value="",
            type="password",
            help="Enter your Anthropic API Key (will be saved to .env file)",
            placeholder="Enter API key" if not current_anthropic_key else "Key already saved"
        )
        
        # Save/Delete buttons for Anthropic
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Anthropic Key", key="save_anthropic"):
                if anthropic_key:
                    if utils.save_api_key("anthropic", anthropic_key):
                        st.session_state['anthropic_api_key'] = anthropic_key
                        st.success("Anthropic API key saved to .env file")
                        st.rerun()
                    else:
                        st.error("Failed to save Anthropic API key")
                else:
                    st.warning("Please enter an API key to save")
        
        with col2:
            if st.button("Delete Anthropic Key", key="delete_anthropic"):
                if utils.delete_api_key("anthropic"):
                    if 'anthropic_api_key' in st.session_state:
                        del st.session_state['anthropic_api_key']
                    st.success("Anthropic API key deleted")
                    st.rerun()
                else:
                    st.warning("No Anthropic API key to delete")
        
        st.markdown("---")
        
        # Mistral settings
        st.subheader("Mistral Settings")
        # Get existing key if available, otherwise empty string
        current_mistral_key = utils.get_api_key("mistral") or ""
        mistral_key_masked = "********" if current_mistral_key else ""
        
        # Display if key is already set
        if current_mistral_key:
            st.success("Mistral API key is set")
        
        # Mistral API Key input
        mistral_key = st.text_input(
            "Mistral API Key", 
            value="",
            type="password",
            help="Enter your Mistral API Key (will be saved to .env file)",
            placeholder="Enter API key" if not current_mistral_key else "Key already saved"
        )
        
        # Save/Delete buttons for Mistral
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Mistral Key", key="save_mistral"):
                if mistral_key:
                    if utils.save_api_key("mistral", mistral_key):
                        st.session_state['mistral_api_key'] = mistral_key
                        st.success("Mistral API key saved to .env file")
                        st.rerun()
                    else:
                        st.error("Failed to save Mistral API key")
                else:
                    st.warning("Please enter an API key to save")
        
        with col2:
            if st.button("Delete Mistral Key", key="delete_mistral"):
                if utils.delete_api_key("mistral"):
                    if 'mistral_api_key' in st.session_state:
                        del st.session_state['mistral_api_key']
                    st.success("Mistral API key deleted")
                    st.rerun()
                else:
                    st.warning("No Mistral API key to delete")
        
        st.markdown("---")
        
        # Ollama settings
        st.subheader("Ollama Settings")
        # Get existing host if available, otherwise default
        current_ollama_host = utils.get_api_key("ollama_host") or "http://localhost:11434"
        
        # Display if host is already set
        st.info(f"Ollama host is set to: {current_ollama_host}")
        
        # Ollama host input
        ollama_host = st.text_input(
            "Ollama Host URL", 
            value="",
            help="Enter your Ollama host URL (will be saved to .env file)",
            placeholder="Enter host URL (e.g., http://localhost:11434)" if current_ollama_host == "http://localhost:11434" else "Host already saved"
        )
        
        # Save/Reset buttons for Ollama host
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Ollama Host", key="save_ollama_host"):
                if ollama_host:
                    if utils.save_api_key("ollama_host", ollama_host):
                        st.session_state['ollama_host'] = ollama_host
                        # Update environment variable immediately
                        os.environ["OLLAMA_HOST"] = ollama_host
                        st.success("Ollama host saved to .env file")
                        st.rerun()
                    else:
                        st.error("Failed to save Ollama host")
                else:
                    st.warning("Please enter a host URL to save")
        
        with col2:
            if st.button("Reset to Default", key="reset_ollama_host"):
                default_host = "http://localhost:11434"
                if utils.save_api_key("ollama_host", default_host):
                    st.session_state['ollama_host'] = default_host
                    # Update environment variable immediately
                    os.environ["OLLAMA_HOST"] = default_host
                    st.success("Ollama host reset to default")
                    st.rerun()
                else:
                    st.warning("Failed to reset Ollama host")
        
        # Display available Ollama models if we can connect
        try:
            from model_clients import list_available_models
            available_models = list_available_models("ollama").get("ollama", [])
            
            if available_models:
                st.success(f"Connected to Ollama! Available models: {', '.join(available_models)}")
            else:
                st.warning("Connected to Ollama, but no models are available. Please pull at least one model.")
        except Exception as e:
            st.error(f"Could not connect to Ollama. Make sure the Ollama service is running at {current_ollama_host}. Error: {str(e)}")
            st.info("To use Ollama, you need to install and run the Ollama service separately. Visit https://ollama.com for installation instructions.")
            
        st.markdown("---")
        
        # Web Scraping Tools API Settings
        st.subheader("Web Scraping API Settings")
        st.caption("Configure API keys for advanced web scraping tools")
        
        # ScrapingBee settings
        st.write("**ScrapingBee**")
        st.caption("Premium web scraping service that handles JavaScript, CAPTCHAs, and IP rotation")
        
        # Get existing key if available
        current_scrapingbee_key = utils.get_api_key("scrapingbee") or ""
        scrapingbee_key_masked = "********" if current_scrapingbee_key else ""
        
        # Display if key is already set
        if current_scrapingbee_key:
            st.success("ScrapingBee API key is set")
        
        # ScrapingBee API Key input
        scrapingbee_key = st.text_input(
            "ScrapingBee API Key", 
            value="",
            type="password",
            help="Enter your ScrapingBee API Key for premium web scraping capabilities",
            placeholder="Enter API key" if not current_scrapingbee_key else "Key already saved",
            key="scrapingbee_api_key_input"
        )
        
        # Save/Delete buttons for ScrapingBee
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save ScrapingBee Key", key="save_scrapingbee"):
                if scrapingbee_key:
                    if utils.save_api_key("scrapingbee", scrapingbee_key):
                        st.session_state['scrapingbee_api_key'] = scrapingbee_key
                        os.environ["SCRAPINGBEE_API_KEY"] = scrapingbee_key
                        st.success("ScrapingBee API key saved to .env file")
                        st.rerun()
                    else:
                        st.error("Failed to save ScrapingBee API key")
        
        st.markdown("---")
        
        # Apify settings
        st.write("**Apify Website Content Crawler**")
        st.caption("Advanced website crawler for extracting content from entire websites at scale")
        
        # Get existing key if available
        current_apify_key = utils.get_api_key("apify") or ""
        apify_key_masked = "********" if current_apify_key else ""
        
        # Display if key is already set
        if current_apify_key:
            st.success("Apify API key is set")
        
        # Apify API Key input
        apify_key = st.text_input(
            "Apify API Key", 
            value="",
            type="password",
            help="Enter your Apify API Key for advanced website crawling",
            placeholder="Enter API key" if not current_apify_key else "Key already saved",
            key="apify_api_key_input"
        )
        
        # Save/Delete buttons for Apify
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Apify Key", key="save_apify"):
                if apify_key:
                    if utils.save_api_key("apify", apify_key):
                        st.session_state['apify_api_key'] = apify_key
                        os.environ["APIFY_API_KEY"] = apify_key
                        st.success("Apify API key saved to .env file")
                        st.rerun()
                    else:
                        st.error("Failed to save Apify API key")
                else:
                    st.warning("Please enter an API key to save")
        
        with col2:
            if st.button("Delete Apify Key", key="delete_apify"):
                if utils.delete_api_key("apify"):
                    if 'apify_api_key' in st.session_state:
                        del st.session_state['apify_api_key']
                    if "APIFY_API_KEY" in os.environ:
                        del os.environ["APIFY_API_KEY"]
                    st.success("Apify API key deleted")
                    st.rerun()
                else:
                    st.error("Failed to delete Apify API key")
        
        # Add ScrapingBee delete button
        with col2:
            if st.button("Delete ScrapingBee Key", key="delete_scrapingbee"):
                if utils.delete_api_key("scrapingbee"):
                    if 'scrapingbee_api_key' in st.session_state:
                        del st.session_state['scrapingbee_api_key']
                    if 'SCRAPINGBEE_API_KEY' in os.environ:
                        del os.environ['SCRAPINGBEE_API_KEY']
                    st.success("ScrapingBee API key deleted")
                    st.rerun()
                else:
                    st.warning("No ScrapingBee API key to delete")
        
        st.markdown("---")
        
        # Custom LLM API settings
        st.subheader("Custom LLM API Settings")
        st.caption("Connect to a custom LLM API endpoint (OpenAI or Anthropic compatible)")
        
        # Get existing values if available
        current_custom_endpoint = utils.get_api_key("custom_api_endpoint") or ""
        current_custom_key = utils.get_api_key("custom_api_key") or ""
        
        # Display if values are already set
        if current_custom_endpoint:
            st.success(f"Custom LLM API endpoint is set: {current_custom_endpoint}")
        
        if current_custom_key:
            st.success("Custom LLM API key is set")
        
        # Custom LLM API endpoint input
        custom_endpoint = st.text_input(
            "Custom LLM API Endpoint", 
            value="",
            help="Enter your custom LLM API endpoint URL (will be saved to .env file)",
            placeholder="Enter API URL (e.g., https://api.yourllm.com/v1/chat/completions)" if not current_custom_endpoint else "Endpoint already saved"
        )
        
        # Custom LLM API Key input (optional)
        custom_api_key = st.text_input(
            "Custom LLM API Key (Optional)", 
            value="",
            type="password",
            help="Enter your custom LLM API Key if required (will be saved to .env file)",
            placeholder="Enter API key if needed" if not current_custom_key else "Key already saved"
        )
        
        # Model name input
        custom_model_name = st.text_input(
            "Custom Model Name",
            value=st.session_state.get("custom_model_name", "custom-model"),
            help="Specify the model name to use with this custom endpoint"
        )
        
        if custom_model_name:
            st.session_state["custom_model_name"] = custom_model_name
        
        # Save/Delete buttons for custom endpoint
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Custom API Settings", key="save_custom_api"):
                saved_successfully = True
                
                # Save endpoint
                if custom_endpoint:
                    if not utils.save_api_key("custom_api_endpoint", custom_endpoint):
                        st.error("Failed to save custom endpoint")
                        saved_successfully = False
                    else:
                        st.session_state["custom_api_endpoint"] = custom_endpoint
                
                # Save API key (if provided)
                if custom_api_key:
                    if not utils.save_api_key("custom_api_key", custom_api_key):
                        st.error("Failed to save custom API key")
                        saved_successfully = False
                    else:
                        st.session_state["custom_api_key"] = custom_api_key
                
                if saved_successfully and (custom_endpoint or custom_api_key):
                    st.success("Custom LLM API settings saved")
                    st.rerun()
                elif not custom_endpoint and not custom_api_key:
                    st.warning("Please enter at least the API endpoint to save")
        
        with col2:
            if st.button("Delete Custom API Settings", key="delete_custom_api"):
                # Delete both endpoint and key
                endpoint_deleted = utils.delete_api_key("custom_api_endpoint")
                key_deleted = utils.delete_api_key("custom_api_key")
                
                if endpoint_deleted or key_deleted:
                    # Clean up session state
                    if "custom_api_endpoint" in st.session_state:
                        del st.session_state["custom_api_endpoint"]
                    if "custom_api_key" in st.session_state:
                        del st.session_state["custom_api_key"]
                    
                    st.success("Custom LLM API settings deleted")
                    st.rerun()
                else:
                    st.warning("No custom API settings to delete")
        
        st.caption("API keys are saved to .env file for persistence between sessions")
    
    # Initialize connection_type with a default value
    connection_type = "Use environment variables"
    
    # Only show database connection for database data source
    if top_level_data_source == "Database":
        st.header("Database Connection")
        
        # Database connection settings
        st.subheader("Connection Settings")
        
        # Connection type selection
        connection_type = st.radio(
            "Connection Type",
            ["Use environment variables", "Custom connection"],
            help="Choose whether to use the built-in database or connect to your own database"
        )
    
    # Custom connection form
    custom_params = {}
    
    # Only process custom connection if we're in database mode
    if st.session_state.data_source_mode in ["tables", "reports"] and connection_type == "Custom connection":
        with st.expander("Database Connection Details", expanded=not st.session_state.connected):
            # Database type selection
            custom_params["db_type"] = st.selectbox(
                "Database Type",
                ["PostgreSQL", "MySQL"],
                index=0,
                help="Choose the database type you want to connect to"
            )
            
            # Convert to lowercase for internal processing
            db_type_lower = custom_params["db_type"].lower()
            
            host_col, port_col = st.columns(2)
            with host_col:
                custom_params["host"] = st.text_input("Host", value="localhost", help="Database server hostname or IP address")
            with port_col:
                # Default port based on database type
                default_port = "5432" if db_type_lower == "postgresql" else "3306"
                custom_params["port"] = st.text_input("Port", value=default_port, help="Database server port")
                
            db_col, user_col = st.columns(2)
            with db_col:
                custom_params["database"] = st.text_input("Database", help="Database name")
            with user_col:
                custom_params["user"] = st.text_input("Username", help="Database username")
                
            custom_params["password"] = st.text_input("Password", type="password", help="Database password")
            
            # SSL options based on database type
            if db_type_lower == "postgresql":
                custom_params["sslmode"] = st.selectbox(
                    "SSL Mode", 
                    ["prefer", "require", "disable", "verify-ca", "verify-full"],
                    index=0,
                    help="SSL connection mode for PostgreSQL"
                )
            else:  # MySQL
                ssl_enabled = st.checkbox("Enable SSL", value=False, help="Enable SSL/TLS for MySQL connection")
                if ssl_enabled:
                    custom_params["ssl_ca"] = st.text_input("SSL CA Certificate Path", help="Path to SSL CA certificate file")
                    custom_params["ssl_verify"] = st.checkbox("Verify SSL Certificate", value=True, help="Verify SSL certificate authenticity")
                else:
                    custom_params["ssl_ca"] = None
                    custom_params["ssl_verify"] = False
    
    # Database connection status and buttons
    if st.session_state.connected:
        st.success("Connected to database")
        if st.button("Disconnect"):
            if st.session_state.conn:
                st.session_state.conn.close()
            st.session_state.connected = False
            st.session_state.conn = None
            st.session_state.tables = []
            st.session_state.tables_info = {}
            st.session_state.selected_tables = []
            st.rerun()
    else:
        st.warning("Not connected to database")
        if st.button("Connect"):
            with st.spinner("Connecting to database..."):
                if connection_type == "Use environment variables":
                    connection = database.get_db_connection()
                else:
                    # Get database type
                    db_type = custom_params.get("db_type", "PostgreSQL").lower()
                    
                    # Common parameters for both database types
                    conn_params = {
                        "host": custom_params.get("host"),
                        "port": custom_params.get("port"),
                        "database": custom_params.get("database"),
                        "user": custom_params.get("user"),
                        "password": custom_params.get("password"),
                        "db_type": db_type
                    }
                    
                    # Add database-specific parameters
                    if db_type == "postgresql":
                        conn_params["sslmode"] = custom_params.get("sslmode")
                    elif db_type == "mysql":
                        conn_params["ssl_ca"] = custom_params.get("ssl_ca")
                        conn_params["ssl_verify"] = custom_params.get("ssl_verify", False)
                    
                    # Create connection
                    connection = database.get_db_connection_with_params(**conn_params)
                
                if connection:
                    st.session_state.conn = connection
                    st.session_state.connected = True
                    
                    # Fetch tables
                    st.session_state.tables = database.get_all_tables(connection)
                    
                    # Get schema for each table
                    st.session_state.tables_info = {}
                    for table in st.session_state.tables:
                        schema = database.get_table_schema(connection, table)
                        st.session_state.tables_info[table] = schema
                    
                    st.success(f"Connected successfully! Found {len(st.session_state.tables)} tables.")
                    st.rerun()
    
    # This code block is for handling mode changes based on the top-level data source selection
    # Check if mode has changed and reset results if needed
    previous_mode = st.session_state.data_source_mode
            
    # Check if mode has changed and reset results if needed
    if previous_mode != new_mode:
        st.session_state.current_results = pd.DataFrame()
        st.session_state.current_query = ""
        st.session_state.current_sql = ""
        st.session_state.web_results = pd.DataFrame()
        
        # Additional resets based on the specific mode
        if new_mode == "web":
            # Reset database and PDF-related selections when switching to web mode
            st.session_state.selected_tables = []
            st.session_state.selected_reports = []
            st.session_state.selected_pdf_collections = []
        elif new_mode == "tables":
            # Reset web, reports, and PDF related selections when switching to tables mode
            st.session_state.selected_collection = None
            st.session_state.selected_reports = []
            st.session_state.selected_pdf_collections = []
        elif new_mode == "reports":
            # Reset web, tables, and PDF related selections when switching to reports mode
            st.session_state.selected_collection = None
            st.session_state.selected_tables = []
            st.session_state.selected_pdf_collections = []
        elif new_mode == "pdf":
            # Reset web, tables, and reports related selections when switching to PDF mode
            st.session_state.selected_collection = None
            st.session_state.selected_tables = []
            st.session_state.selected_reports = []
        elif new_mode == "image":
            # Reset web, tables, reports, and PDF related selections when switching to Image Analysis mode
            st.session_state.selected_collection = None
            st.session_state.selected_tables = []
            st.session_state.selected_reports = []
            st.session_state.selected_pdf_collections = []
    
    # Update the mode
    st.session_state.data_source_mode = new_mode
        
    # If connected to a database, show source selection
    if top_level_data_source == "Database" and hasattr(st.session_state, 'conn') and st.session_state.conn:
        st.subheader("Choose Database Source")
        
        # Get reverse mapping for current mode to radio options
        reverse_mapping = {
            "tables": "Database Tables",
            "reports": "Custom Reports"
        }
        
        # Get current display name for the radio button
        current_display = reverse_mapping.get(st.session_state.data_source_mode, "Database Tables")
        
        # Show sub-selection for database
        selected_display = st.radio(
            "Database source",
            ["Database Tables", "Custom Reports"],
            index=0 if current_display == "Database Tables" else 1,
            help="Select which database source to interact with",
            horizontal=True
        )
        
        # Map radio options to mode values
        mode_mapping = {
            "Database Tables": "tables",
            "Custom Reports": "reports"
        }
        
        # Get the new mode from the mapping
        new_mode = mode_mapping[selected_display]
        
        # Update if changed
        if st.session_state.data_source_mode != new_mode:
            st.session_state.data_source_mode = new_mode
            st.rerun()
        
        st.markdown("---")
    
    # Database Tables mode
    if st.session_state.data_source_mode == "tables":
        st.header("Database Tables")
        
        # Tables selection
        if st.session_state.tables:
            # Add "Select All Tables" option
            col1, col2 = st.columns([3, 1])
            with col1:
                select_all = st.checkbox("Select All Tables", key="select_all_tables")
            
            # If "Select All" is checked, use all tables as the default
            if select_all:
                default_selection = st.session_state.tables
            else:
                default_selection = st.session_state.selected_tables
            
            # Tables multiselect without immediate rerun
            selected = st.multiselect(
                "Select tables to query",
                options=st.session_state.tables,
                default=default_selection
            )
            
            # Apply button to confirm selection
            apply_selection = st.button("Apply Selection")
            
            # Only update and rerun when Apply button is clicked
            if apply_selection and selected != st.session_state.selected_tables:
                st.session_state.selected_tables = selected
                st.rerun()
            
            # Show table details if selected
            if st.session_state.selected_tables:
                table_counts = database.get_table_counts(st.session_state.conn, st.session_state.selected_tables)
                
                st.subheader("Table Information")
                for table in st.session_state.selected_tables:
                    with st.expander(f"{table} ({table_counts.get(table, 'Unknown')} rows)"):
                        st.dataframe(st.session_state.tables_info[table], use_container_width=True)
                        
                        # Show sample data
                        st.subheader("Sample Data")
                        sample = database.get_table_sample(st.session_state.conn, table)
                        st.dataframe(sample, use_container_width=True)
        else:
            st.info("No tables found in the database")
                
    # Custom Reports mode
    elif st.session_state.data_source_mode == "reports":
        st.header("Custom Reports")
        
        # Add custom report
        with st.expander("Create Custom Report", expanded=False):
            report_name = st.text_input("Report Name", key="new_report_name")
            report_description = st.text_area("Report Description", key="new_report_desc")
            report_query = st.text_area("SQL Query", height=200, key="new_report_query", 
                                      help="Enter a valid SQL query to create a custom report")
            
            if st.button("Create Report", key="btn_create_report"):
                if st.session_state.conn:
                    success, message = reports.add_custom_report(report_name, report_description, report_query)
                    
                    if success:
                        st.success(message)
                        # Update available reports list
                        st.session_state.available_reports = reports.get_report_list()
                        # Use a flag in session state to trigger a reset after rerun
                        st.session_state.reset_report_form = True
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please connect to a database first")
            
        # Manage existing reports
        with st.expander("Manage Reports", expanded=False):
            # Initialize reports
            reports.initialize_reports()
            
            # Reports list for deletion
            report_list_for_deletion = st.session_state.available_reports
            
            if report_list_for_deletion:
                report_to_delete = st.selectbox(
                    "Select Report to Delete",
                    options=report_list_for_deletion,
                    key="report_to_delete"
                )
                
                if st.button("Delete Report", key="btn_delete_report"):
                    if report_to_delete:
                        success, message = reports.delete_custom_report(report_to_delete)
                        
                        if success:
                            st.success(message)
                            # Update available reports list
                            st.session_state.available_reports = reports.get_report_list()
                            # Check if the deleted report was selected
                            if report_to_delete in st.session_state.selected_reports:
                                st.session_state.selected_reports.remove(report_to_delete)
                            st.rerun()
                        else:
                            st.error(message)
            else:
                st.info("No custom reports available to delete")
            
            # Reports selection
            report_list = st.session_state.available_reports
            
            if report_list:
                selected_reports = st.multiselect(
                    "Select reports to query",
                    options=report_list,
                    default=st.session_state.selected_reports
                )
                
                if selected_reports != st.session_state.selected_reports:
                    # Update selected reports
                    st.session_state.selected_reports = selected_reports
                    
                    # Get report schemas
                    st.session_state.reports_info = {}
                    for report in selected_reports:
                        # Generate schema for the report
                        schema = reports.get_report_schema(report)
                        st.session_state.reports_info[report] = schema
                    
                    st.rerun()
                
                # Show report details if selected
                if st.session_state.selected_reports:
                    st.subheader("Report Information")
                    
                    for report in st.session_state.selected_reports:
                        with st.expander(f"{report}", expanded=False):
                            # Display description
                            st.markdown(f"**Description:** {reports.get_report_description(report)}")
                            
                            # Display schema if available
                            if report in st.session_state.reports_info:
                                st.subheader("Expected Columns")
                                st.dataframe(st.session_state.reports_info[report], use_container_width=True)
                            
                            # Show sample data by executing the report
                            st.subheader("Sample Results")
                            try:
                                with st.spinner(f"Loading {report} data..."):
                                    result = reports.execute_report(st.session_state.conn, report)
                                    
                                    # Handle both tuple and non-tuple return formats
                                    if isinstance(result, tuple):
                                        sample, error_message, _ = result
                                    else:
                                        sample, error_message = result
                                        
                                    if error_message:
                                        st.error(error_message)
                                        # Show the SQL query for troubleshooting
                                        st.subheader("Report SQL Query")
                                        st.code(reports.get_report_query(report), language="sql")
                                        st.info("Tip: You may need to update this report's SQL query to match your database schema.")
                                    elif not sample.empty:
                                        st.dataframe(sample.head(5), use_container_width=True)
                                    else:
                                        st.info("The report executed successfully but returned no data.")
                            except Exception as e:
                                st.error(f"Error loading report data: {str(e)}")
            else:
                st.info("No custom reports available")
                
    # Web Content mode - Show collection selection and management options in sidebar
    elif st.session_state.data_source_mode == "web":
        st.header("Website Content Configuration")
        
        # Refresh collections list
        st.session_state.web_collections = web_scraper.get_all_collections()
        
        # Collections selection
        if st.session_state.web_collections:
            st.subheader("Step 1: Select Website Collection(s)")
            
            # Use multi-select mode by default - similar to database tables selection
            # Add "Select All Collections" option
            col1, col2 = st.columns([3, 1])
            with col1:
                select_all = st.checkbox("Select All Collections", key="select_all_web_collections")
            
            # If "Select All" is checked, use all collections as the default
            if select_all:
                default_selection = st.session_state.web_collections
            else:
                default_selection = st.session_state.selected_collections
                
            # Multi-select mode without immediate rerun (like database table selection)
            selected_collections = st.multiselect(
                "Select collections to query",
                options=st.session_state.web_collections,
                default=default_selection
            )
            
            # Apply button to confirm selection (matching database table selection behavior)
            apply_selection = st.button("Apply Selection", key="apply_web_selection")
            
            # Only update and rerun when Apply button is clicked
            if apply_selection and selected_collections != st.session_state.selected_collections:
                st.session_state.selected_collections = selected_collections
                # For backwards compatibility
                st.session_state.selected_collection = selected_collections[0] if selected_collections else None
                # Get collection statistics for the first selected collection
                if selected_collections:
                    st.session_state.collection_stats = web_scraper.get_collection_stats(selected_collections[0])
                st.rerun()
            
            # Determine if we're in multi-select mode based on whether there are selected collections
            multi_select_mode = len(st.session_state.selected_collections) > 0
            
            # Display info about multiple collections
            if multi_select_mode:
                st.info(f"You have selected {len(st.session_state.selected_collections)} collections to query simultaneously.")
            else:
                # Single select mode (original behavior)
                selected_collection = st.selectbox(
                    "Select a collection to query",
                    options=st.session_state.web_collections,
                    index=st.session_state.web_collections.index(st.session_state.selected_collection) if st.session_state.selected_collection in st.session_state.web_collections else 0
                )
                
                if selected_collection != st.session_state.selected_collection:
                    st.session_state.selected_collection = selected_collection
                    # Clear multi-select list when using single select
                    st.session_state.selected_collections = []
                    # Get collection statistics
                    st.session_state.collection_stats = web_scraper.get_collection_stats(selected_collection)
                    st.rerun()
            
            # Display collection stats only for single collection mode
            if not multi_select_mode and st.session_state.selected_collection:
                stats = web_scraper.get_collection_stats(st.session_state.selected_collection)
                
                st.subheader("Collection Information")
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Documents", stats.get("document_count", 0))
                
                with stat_col2:
                    st.metric("Sources", len(stats.get("sources", [])))
                
                if stats.get("sources"):
                    with st.expander("Source URLs", expanded=False):
                        for source in stats.get("sources", []):
                            st.markdown(f"- [{source}]({source})")
                            
                st.markdown("---")
            else:
                st.info("No collections available. Add a new website below.")
            
            # Web scraping functionality with better layout
            st.subheader("Step 2: Add New Website")
            
            url = st.text_input(
                "Website URL", 
                placeholder="https://example.com",
                help="Enter the URL of the website you want to scrape",
                key="web_url_sidebar"
            )
            
            collection_name = st.text_input(
                "Collection Name",
                placeholder="my-website-collection",
                help="Enter a name for this collection (use only letters, numbers, and hyphens)",
                key="collection_name_sidebar"
            )
            
            # Options for advanced scraping
            with st.expander("Advanced Scraping Options", expanded=False):
                # Crawling Mode Section
                st.markdown("### Crawling Mode")
                crawl_mode = st.radio(
                    "Select Crawling Mode",
                    options=["Single Page", "Recursive Crawling"],
                    index=0,
                    horizontal=True,
                    help="Single Page: Extract content from just the URL provided. Recursive Crawling: Follow links and collect content from multiple pages."
                )
                
                recursive = crawl_mode == "Recursive Crawling"
                use_apify = False
                
                if recursive:
                    # Check if Apify API key is available
                    apify_api_key = utils.get_api_key("apify")
                    
                    # Crawler Engine Selection
                    crawler_engine = st.radio(
                        "Crawler Engine",
                        options=["Standard Crawler", "Apify Website Content Crawler"],
                        index=0,
                        horizontal=True,
                        help="Standard Crawler: Built-in crawler with ScrapingBee support. Apify: Professional web crawler for comprehensive site extraction."
                    )
                    
                    use_apify = crawler_engine == "Apify Website Content Crawler"
                    
                    if use_apify:
                        if not apify_api_key:
                            st.warning("Apify API key not found. Please add it in the Model API Settings. Falling back to standard crawler.")
                            use_apify = False
                        else:
                            st.success("Using Apify Website Content Crawler for advanced crawling capabilities")
                            st.info("Apify can handle complex websites with JavaScript, dynamic content, and pagination while respecting robots.txt rules.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        max_pages = st.number_input(
                            "Maximum Pages to Crawl",
                            min_value=1, 
                            max_value=50 if not use_apify else 200,  # Higher limit for Apify
                            value=10,
                            help="Maximum number of pages to crawl when using recursive mode"
                        )
                    
                    with col2:
                        same_domain_only = st.checkbox(
                            "Stay on Same Domain Only", 
                            value=True,
                            help="When enabled, only follows links that are on the same domain as the starting URL"
                        )
                    
                    if use_apify:
                        max_crawl_depth = st.slider(
                            "Maximum Crawl Depth",
                            min_value=1,
                            max_value=10,
                            value=5,
                            help="How many links deep the crawler will go from the starting URL"
                        )
                        st.info("Apify crawler will extract content from up to " + str(max_pages) + " pages, going up to " + str(max_crawl_depth) + " links deep from the starting URL. This comprehensive crawling process may take longer but provides better content coverage.")
                    else:
                        # Default value for standard crawler
                        max_crawl_depth = 5
                        st.info("Recursive crawling will follow links from the start URL and crawl multiple pages automatically. This process will take longer depending on how many pages you set as the maximum.")
                else:
                    # Set defaults for single page mode
                    max_pages = 1
                    same_domain_only = True
                    use_apify = False
                    max_crawl_depth = 1
                
                st.markdown("---")
                st.markdown("### Scraping Engine")
                
                # Check if ScrapingBee API key is available
                scrapingbee_api_key = utils.get_api_key("scrapingbee")
                
                if scrapingbee_api_key:
                    st.success("Premium scraping enabled with ScrapingBee! 🐝")
                    
                    use_premium = st.checkbox(
                        "Use Premium Scraper (ScrapingBee)", 
                        value=True,
                        help="Use ScrapingBee premium service for websites with strong anti-scraping protections",
                        key="use_premium_sidebar"
                    )
                    
                    if use_premium:
                        st.info("Premium mode uses ScrapingBee to handle websites with JavaScript, CAPTCHAs, and anti-bot protection. It's more reliable for complex websites.")
                        
                        # ScrapingBee-specific options
                        premium_render_js = st.checkbox(
                            "Render JavaScript", 
                            value=True,
                            help="Enable JavaScript rendering for dynamic websites",
                            key="premium_render_js"
                        )
                        
                        premium_country = st.selectbox(
                            "Geolocation",
                            ["", "us", "gb", "fr", "de", "jp", "ca", "es", "it", "br", "au", "in"],
                            format_func=lambda x: "Default" if x == "" else x.upper(),
                            help="Select country to access geo-restricted content",
                            key="premium_country"
                        )
                        
                        # Use Selenium checkbox (disabled when premium is enabled)
                        use_selenium = False
                        st.checkbox(
                            "Use Selenium", 
                            value=False,
                            help="Disabled when premium scraping is enabled",
                            disabled=True,
                            key="use_selenium_sidebar_disabled"
                        )
                    else:
                        # If premium is available but not selected, show Selenium option
                        use_selenium = st.checkbox(
                            "Use Selenium (better for complex websites)", 
                            value=False,
                            help="Enable this for websites with anti-scraping protections or dynamic content",
                            key="use_selenium_sidebar"
                        )
                        
                        if use_selenium:
                            st.info("Selenium mode uses a headless browser to better handle websites with anti-scraping protections. This may take longer but is more likely to succeed with complex websites.")
                else:
                    # No ScrapingBee key available, show only Selenium option
                    use_premium = False
                    premium_render_js = True  # Default values
                    premium_country = ""      # Default values
                    
                    use_selenium = st.checkbox(
                        "Use Selenium (better for complex websites)", 
                        value=False,
                        help="Enable this for websites with anti-scraping protections or dynamic content",
                        key="use_selenium_sidebar"
                    )
                    
                    if use_selenium:
                        st.info("Selenium mode uses a headless browser to better handle websites with anti-scraping protections. This may take longer but is more likely to succeed with complex websites.")
                    
                    st.warning("For advanced scraping capabilities, add a ScrapingBee API key in Model API Settings.")
                    st.markdown("[Get a ScrapingBee API key](https://www.scrapingbee.com/) for premium web scraping capabilities including JavaScript rendering, CAPTCHA bypass, and IP rotation.")
            
            if st.button("Scrape Website", key="scrape_btn_sidebar", type="primary"):
                if url and collection_name:
                    # Display a different message for recursive mode to set expectations
                    spinner_message = f"Scraping content from {url}..." + (" (Recursive mode - this may take a while)" if recursive else "")
                    with st.spinner(spinner_message):
                        try:
                            # Add metadata about when this was scraped and the crawl mode
                            metadata = {
                                "scraped_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "crawl_mode": "recursive" if recursive else "single_page"
                            }
                            
                            # Scrape website with the new parameters
                            result = web_scraper.scrape_website(
                                url=url,
                                collection_name=collection_name,
                                use_selenium=use_selenium,
                                use_premium=use_premium if 'use_premium' in locals() else False,
                                premium_render_js=premium_render_js if 'premium_render_js' in locals() else True,
                                premium_country=premium_country if 'premium_country' in locals() else "",
                                recursive=recursive,
                                max_pages=max_pages if recursive else 1,
                                max_crawl_depth=max_crawl_depth if 'max_crawl_depth' in locals() else 5,
                                same_domain_only=same_domain_only if recursive else True,
                                use_apify=use_apify if 'use_apify' in locals() else False,
                                metadata=metadata
                            )
                            
                            if result and result.get("success"):
                                # For recursive crawling, show additional info on pages crawled
                                if recursive and "pages_crawled" in result:
                                    st.success(f"Successfully scraped {result.get('chunk_count')} chunks from {result.get('pages_crawled')} pages starting at {url}")
                                    # Show a sample of the URLs crawled if available
                                    if "urls_crawled" in result and len(result["urls_crawled"]) > 0:
                                        with st.expander("Pages Crawled"):
                                            # Show up to 10 URLs in the UI to avoid cluttering
                                            urls_to_show = result["urls_crawled"][:10]
                                            for crawled_url in urls_to_show:
                                                st.write(f"- {crawled_url}")
                                            if len(result["urls_crawled"]) > 10:
                                                st.write(f"... and {len(result['urls_crawled']) - 10} more pages")
                                else:
                                    # Standard single-page message
                                    st.success(f"Successfully scraped {result.get('chunk_count')} chunks from {url}")
                                
                                # Update session state
                                st.session_state.selected_collection = collection_name
                                st.session_state.web_collections = web_scraper.get_all_collections()
                                st.rerun()
                            else:
                                st.error(f"Failed to scrape website: {result.get('error')}")
                        except Exception as e:
                            st.error(f"Error scraping website: {str(e)}")
                else:
                    st.warning("Please enter both a URL and collection name.")
                    
            # Add a separator            
            st.markdown("---")
            
            # Collection management
            st.subheader("Manage Website Collections")
            
            if st.session_state.web_collections:
                collection_to_delete = st.selectbox(
                    "Select Collection to Delete",
                    options=st.session_state.web_collections,
                    key="collection_to_delete_sidebar"
                )
                
                if st.button("Delete Collection", key="btn_delete_collection_sidebar", type="secondary"):
                    if collection_to_delete:
                        success = web_scraper.delete_collection(collection_to_delete)
                        
                        if success:
                            st.success(f"Successfully deleted collection: {collection_to_delete}")
                            # Check if the deleted collection was selected
                            if collection_to_delete == st.session_state.selected_collection:
                                st.session_state.selected_collection = None
                            # Refresh collections list
                            st.session_state.web_collections = web_scraper.get_all_collections()
                            st.rerun()
                        else:
                            st.error(f"Failed to delete collection: {collection_to_delete}")
            else:
                st.info("No collections available to delete")
                
            # Add a message to direct users to the main panel
            st.markdown("---")
            st.info("👉 After setting up your website collection, go to the main panel to ask questions about the website content.")
        
    # PDF Documents mode
    elif st.session_state.data_source_mode == "pdf":
        st.header("PDF Documents Configuration")
        
        # Import the PDF processor
        from pdf_processor import PDFProcessor, find_collections
        
        # Refresh collections list
        pdf_processor = PDFProcessor(st.session_state.model_provider, st.session_state.current_model)
        # Store the PDF processor in session state so it can be accessed elsewhere
        st.session_state.pdf_processor = pdf_processor
        pdf_collections = find_collections()
        
        # Update session state with metadata from PDF processor
        # This ensures we're getting the persisted metadata with filenames
        st.session_state.pdf_collections = {}
        for collection_id in pdf_collections:
            metadata = st.session_state.pdf_processor.get_pdf_metadata(collection_id)
            if metadata:
                st.session_state.pdf_collections[collection_id] = metadata
        
        # PDF file upload section
        st.subheader("Step 1: Upload PDF Document")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF file", 
            type=["pdf"],
            help="Select a PDF file to upload and analyze",
            key="pdf_file_uploader"
        )
        
        if uploaded_file is not None:
            # Display a preview of the uploaded file
            st.write(f"Uploaded: **{uploaded_file.name}**")
            
            # Process button
            if st.button("Process PDF", key="process_pdf_btn", type="primary"):
                try:
                    with st.spinner("Processing PDF..."):
                        # Process the PDF file
                        collection_id = st.session_state.pdf_processor.process_pdf(uploaded_file, uploaded_file.name)
                        
                        if collection_id:
                            st.success(f"Successfully processed PDF: {uploaded_file.name}")
                            
                            # Update the collections in session state
                            pdf_collections = find_collections()
                            # Update session state with metadata from PDF processor
                            st.session_state.pdf_collections = {}
                            for cid in pdf_collections:
                                metadata = st.session_state.pdf_processor.get_pdf_metadata(cid)
                                if metadata:
                                    st.session_state.pdf_collections[cid] = metadata
                            
                            # Select the newly processed document
                            if collection_id not in st.session_state.selected_pdf_collections:
                                st.session_state.selected_pdf_collections.append(collection_id)
                            
                            st.rerun()
                        else:
                            st.error("Failed to process PDF. Please try again with a different file.")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        
        # Add a separator
        st.markdown("---")
        
        # PDF collection selection
        st.subheader("Step 2: Select PDF Documents")
        
        if st.session_state.pdf_collections:
            # Create a more user-friendly selection list with file names
            collection_options = {
                cid: metadata.get("filename", "Unknown") 
                for cid, metadata in st.session_state.pdf_collections.items()
            }
            
            # Add "Select All PDFs" option
            col1, col2 = st.columns([3, 1])
            with col1:
                select_all = st.checkbox("Select All PDFs", key="select_all_pdfs")
            
            # If "Select All" is checked, use all PDFs as the default
            if select_all:
                default_selection = list(collection_options.keys())
            else:
                # Default selection based on session state
                default_selection = []
                for cid in st.session_state.selected_pdf_collections:
                    if cid in collection_options:
                        default_selection.append(cid)
            
            # Display as multiselect without immediate rerun (like database table selection)
            selected_collections = st.multiselect(
                "Select PDF documents to query",
                options=list(collection_options.keys()),
                default=default_selection,
                format_func=lambda cid: collection_options[cid],
                help="Select one or more PDFs to analyze"
            )
            
            # Apply button to confirm selection (matching database table selection behavior)
            apply_selection = st.button("Apply Selection", key="apply_pdf_selection")
            
            # Only update and rerun when Apply button is clicked
            if apply_selection and selected_collections != st.session_state.selected_pdf_collections:
                st.session_state.selected_pdf_collections = selected_collections
                st.rerun()
            
            # Show collection details if selected
            if st.session_state.selected_pdf_collections:
                st.subheader("Selected Documents")
                
                for collection_id in st.session_state.selected_pdf_collections:
                    if collection_id in st.session_state.pdf_collections:
                        metadata = st.session_state.pdf_collections[collection_id]
                        
                        with st.expander(f"{metadata.get('filename', 'Unknown')}"):
                            # Show metadata
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Pages", metadata.get("page_count", "Unknown"))
                            
                            with col2:
                                st.metric("Chunks", metadata.get("chunk_count", "Unknown"))
                            
                            # Display the first page as a preview if available
                            if "has_images" in metadata and metadata["has_images"]:
                                # Get the first page image
                                st.subheader("Document Preview")
                                
                                # Create a more interactive PDF viewer for previewing the document
                                metadata = st.session_state.pdf_collections.get(collection_id, {})
                                pdf_viewer_html = st.session_state.pdf_processor.create_interactive_pdf_viewer(
                                    collection_id=collection_id,
                                    metadata=metadata,
                                    pdf_processor=st.session_state.pdf_processor,
                                    highlight_page=1
                                )
                                
                                # Display the interactive PDF viewer
                                st.markdown(pdf_viewer_html, unsafe_allow_html=True)
                                
                                # Fallback to displaying just the first page image if the viewer fails
                                if "<p>Error creating PDF viewer" in pdf_viewer_html:
                                    first_page_image = st.session_state.pdf_processor.get_page_image(collection_id, 1)
                                    if first_page_image:
                                        st.subheader("First Page Preview (Fallback)")
                                        st.image(f"data:image/jpeg;base64,{first_page_image}", 
                                                use_column_width=True)
        else:
            st.info("No PDF documents available. Upload a PDF file to get started.")
        
        # Add a separator
        st.markdown("---")
        
        # PDF collection management
        st.subheader("Manage PDF Documents")
        
        if st.session_state.pdf_collections:
            # Create a list of options with document names
            delete_options = {
                cid: metadata.get("filename", "Unknown") 
                for cid, metadata in st.session_state.pdf_collections.items()
            }
            
            # Dropdown to select document to delete
            collection_to_delete = st.selectbox(
                "Select PDF to Delete",
                options=list(delete_options.keys()),
                format_func=lambda cid: delete_options[cid],
                key="pdf_to_delete"
            )
            
            if st.button("Delete PDF", key="btn_delete_pdf", type="secondary"):
                if collection_to_delete:
                    try:
                        # Delete the collection
                        success = st.session_state.pdf_processor.delete_collection(collection_to_delete)
                        
                        if success:
                            st.success(f"Successfully deleted PDF: {delete_options[collection_to_delete]}")
                            
                            # Remove from selected collections if it was there
                            if collection_to_delete in st.session_state.selected_pdf_collections:
                                st.session_state.selected_pdf_collections.remove(collection_to_delete)
                            
                            # Update collections
                            pdf_collections = find_collections()
                            # Update session state with metadata from PDF processor
                            st.session_state.pdf_collections = {}
                            for cid in pdf_collections:
                                metadata = st.session_state.pdf_processor.get_pdf_metadata(cid)
                                if metadata:
                                    st.session_state.pdf_collections[cid] = metadata
                            
                            st.rerun()
                        else:
                            st.error(f"Failed to delete PDF: {delete_options[collection_to_delete]}")
                    except Exception as e:
                        st.error(f"Error deleting PDF: {str(e)}")
        else:
            st.info("No PDF documents available to delete")
        
        # Add a message to direct users to the main panel
        st.markdown("---")
        st.info("👉 After selecting PDF documents, go to the main panel to ask questions about their content.")
    
    # Show report details if selected
    if st.session_state.data_source_mode == "reports" and st.session_state.selected_reports:
        st.subheader("Report Information")
        
        for report in st.session_state.selected_reports:
            with st.expander(f"{report}", expanded=False):
                # Display description
                st.markdown(f"**Description:** {reports.get_report_description(report)}")
                
                # Display schema if available
                if report in st.session_state.reports_info:
                    st.subheader("Expected Columns")
                    st.dataframe(st.session_state.reports_info[report], use_container_width=True)
                
                # Show sample data by executing the report
                st.subheader("Sample Results")
                try:
                    with st.spinner(f"Loading {report} data..."):
                        result = reports.execute_report(st.session_state.conn, report)
                        # Handle return values based on length of tuple
                        if len(result) == 3:
                            sample, error_message, _ = result
                        else:
                            sample, error_message = result
                            
                        if error_message:
                            st.error(error_message)
                            # Show the SQL query for troubleshooting
                            st.subheader("Report SQL Query")
                            st.code(reports.get_report_query(report), language="sql")
                            st.info("Tip: You may need to update this report's SQL query to match your database schema.")
                        elif not sample.empty:
                            st.dataframe(sample.head(5), use_container_width=True)
                        else:
                            st.info("The report executed successfully but returned no data.")
                except Exception as e:
                    st.error(f"Error loading report data: {str(e)}")
    # Display message if no custom reports are available
    if st.session_state.data_source_mode == "reports" and not st.session_state.available_reports:
        st.info("No custom reports available")
    # Query history
    if st.session_state.query_history:
        st.header("Query History")
        for i, (query, sql, timestamp) in enumerate(st.session_state.query_history):
            if st.button(f"{timestamp}: {query[:40]}...", key=f"history_{i}"):
                st.session_state.current_query = query
                st.session_state.current_sql = sql
                st.rerun()

# Main content area
# For PDF, Web Content, and Images, we don't need a database connection
if st.session_state.data_source_mode in ["pdf", "web", "image"]:
    # Show the appropriate interface for non-database modes
    if st.session_state.data_source_mode == "web":
        # Simplified web content query interface
        st.header("Ask Questions About Website Content")
        
        if not st.session_state.selected_collection:
            # Show help message for new users
            st.info("👋 Welcome to the Web Content feature! To get started, please select or add a website using the sidebar.")
            
            # Quick explanation
            st.markdown("""
            **How it works:**
            1. Add a website URL to scrape using the sidebar
            2. The system will extract and store the website text
            3. Ask questions about the website content
            4. AI will search and analyze the content to provide answers
            """)
            
            # Disabled query interface
            st.text_area(
                "Enter your question about the website content",
                value="",
                height=100,
                placeholder="Example: What are the main topics discussed on this website?",
                help="Ask questions about the website content in plain English.",
                disabled=True
            )
            
            # Disabled submit button
            st.button("Submit", disabled=True, type="primary")
        else:
            # Regular query interface for users with a collection
            if len(st.session_state.selected_collections) > 0:
                # Multi-collection mode
                st.subheader(f"Querying across {len(st.session_state.selected_collections)} website collections")
                collection_names = ", ".join(st.session_state.selected_collections)
                st.caption(f"Selected collections: {collection_names}")
            else:
                # Single collection mode
                st.subheader(f"Using collection: {st.session_state.selected_collection}")
            
            # User query input
            query = st.text_area(
                "Enter your question about the website content",
                value="",  # Don't initialize with previous query
                key="web_query_input",
                height=100,
                placeholder="Example: What are the main topics discussed on this website?",
                help="Ask questions about the website content in plain English."
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                execute = st.button("Submit", use_container_width=True, type="primary")
            
            with col2:
                clear = st.button("Clear Results", use_container_width=True)
                
            if clear:
                st.session_state.current_query = ""
                st.session_state.web_results = pd.DataFrame()
                st.rerun()
                
            if execute and query:
                st.session_state.current_query = query
                
                with st.spinner("Processing your question..."):
                    try:
                        # Determine which collections to query
                        if len(st.session_state.selected_collections) > 0:
                            # Use multiple collections
                            collection_name = st.session_state.selected_collections
                            st.info(f"Searching across {len(collection_name)} collections...")
                        else:
                            # Use single collection
                            collection_name = str(st.session_state.selected_collection)
                        
                        # Display answer heading
                        st.markdown("### Answer")
                        
                        # Create a checkbox for streaming option (default to true)
                        use_streaming = st.checkbox("Enable streaming response", 
                                                   value=True, 
                                                   help="Show response as it's being generated",
                                                   key="web_streaming_main")
                        
                        # Get AI-generated answer with streaming if selected
                        answer = web_scraper.ask_question_about_website(
                            query,
                            collection_name,
                            model_provider=st.session_state.model_provider,
                            use_streaming=use_streaming
                        )
                        
                        # If not using streaming, we need to display the answer
                        # (streaming version displays itself through the placeholder)
                        if not use_streaming or st.session_state.model_provider != "openai":
                            st.markdown(answer)
                        
                        # Get combined results from all collections
                        combined_results = []
                        if isinstance(collection_name, list):
                            # When using multiple collections
                            for coll in collection_name:
                                results = web_scraper.search_website_content(
                                    query, 
                                    coll,
                                    n_results=3  # Reduce results per collection to avoid overwhelming
                                )
                                if not results.empty:
                                    # Add collection name to results
                                    results['collection'] = coll
                                    combined_results.append(results)
                            
                            # Combine all results if we have any
                            if combined_results:
                                results_df = pd.concat(combined_results, ignore_index=True)
                            else:
                                results_df = pd.DataFrame()
                        else:
                            # Single collection - search as before
                            results_df = web_scraper.search_website_content(
                                query, 
                                collection_name,
                                n_results=5
                            )
                        
                        # Store results in session state
                        st.session_state.web_results = results_df
                        
                        # Display source passages
                        if not results_df.empty:
                            with st.expander("Source Passages", expanded=False):
                                st.markdown("These passages from the website were used to generate the answer:")
                                for i, row in results_df.iterrows():
                                    # Display collection name if available
                                    if 'collection' in row:
                                        collection_name_display = row['collection']
                                        st.markdown(f"**Collection:** {collection_name_display}")
                                    
                                    st.markdown(f"**Source:** [{row['source']}]({row['source']})")
                                    st.markdown(f"{row['content'][:500]}...")
                                    st.markdown("---")
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
    
    elif st.session_state.data_source_mode == "pdf":
        # PDF Documents query interface
        st.header("Ask Questions About PDF Documents")
        
        # Import PDF QA
        from pdf_qa import PDFQuestionAnswering
        
        # No longer need the PDF source viewer that used rerun() - we're now showing PDFs inline with the button click
        
        # Regular PDF Q&A interface
        if not st.session_state.selected_pdf_collections:
            # Show help message for new users
            st.info("👋 Welcome to the PDF Documents feature! To get started, please upload and select PDF documents using the sidebar.")
            
            # Quick explanation
            st.markdown("""
            **How it works:**
            1. Upload PDF documents using the sidebar
            2. Select one or more PDF documents to analyze
            3. Ask questions about the PDF content
            4. AI will search the documents and provide detailed answers with source references
            """)
            
            # Disabled query interface
            st.text_area(
                "Enter your question about the PDF documents",
                value="",
                height=100,
                placeholder="Example: What are the key points in this document?",
                help="Ask questions about the PDF documents in plain English.",
                disabled=True
            )
            
            # Disabled submit button
            st.button("Submit", disabled=True, type="primary")
        else:
            # Get file names for selected collections
            selected_doc_names = []
            for cid in st.session_state.selected_pdf_collections:
                if cid in st.session_state.pdf_collections:
                    selected_doc_names.append(st.session_state.pdf_collections[cid].get("filename", "Unknown"))
            
            # Display selected documents
            st.subheader(f"Using {len(selected_doc_names)} selected document(s)")
            st.write(f"Documents: {', '.join(selected_doc_names)}")
            
            # User query input
            query = st.text_area(
                "Enter your question about the PDF documents",
                value="",  # Don't initialize with previous query
                key="pdf_query_input",
                height=100,
                placeholder="Example: What are the key points in this document?",
                help="Ask questions about the PDF documents in plain English."
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                execute = st.button("Submit", use_container_width=True, type="primary")
            
            with col2:
                clear = st.button("Clear Results", use_container_width=True)
                
            if clear:
                st.session_state.current_query = ""
                st.rerun()
                
            if execute and query:
                st.session_state.current_query = query
                # Store the query in session state for use with PDF highlighting
                st.session_state.query = query
                
                with st.spinner("Processing your question..."):
                    try:
                        # Initialize the PDF Q&A processor
                        pdf_qa = PDFQuestionAnswering(
                            model_provider=st.session_state.model_provider, 
                            model_name=st.session_state.get('current_model')
                        )
                        
                        # Display answer heading
                        st.markdown("### Answer")
                        
                        # Create a checkbox for streaming option (default to true)
                        use_streaming = st.checkbox("Enable streaming response", 
                                                 value=True, 
                                                 help="Show response as it's being generated",
                                                 key="pdf_streaming_main")
                        
                        # Create placeholder for streaming output
                        answer_placeholder = st.empty()
                        
                        # Ask the question about the PDF content with streaming if selected
                        try:
                            if use_streaming:
                                # Get the answer text and sources
                                answer_text = ""
                                result = pdf_qa.stream_answer(
                                    query=query,
                                    collection_ids=st.session_state.selected_pdf_collections,
                                    use_multiquery=True,
                                    streamlit_placeholder=answer_placeholder
                                )
                                # In streaming mode, the answer text is returned via the placeholder
                                sources = result.get("sources", [])
                                # After streaming, the answer_text attribute should be set on the placeholder
                                answer_text = getattr(answer_placeholder, "answer_text", "No response generated")
                            else:
                                # Non-streaming version
                                result = pdf_qa.answer_question(
                                    query=query,
                                    collection_ids=st.session_state.selected_pdf_collections,
                                    use_multiquery=True
                                )
                                
                                # Store the answer text (but don't display it yet - we'll display with source links)
                                answer_text = result["answer"]
                                sources = result.get("sources", [])
                        except Exception as e:
                            error_message = f"Error answering question: {str(e)}"
                            st.error(error_message)
                            answer_placeholder.error(error_message)
                            # Set sources to empty list to avoid errors in subsequent code
                            sources = []
                            
                        # Function to make source references clickable
                        def process_source_references(text, sources):
                            """
                            Replace [Source X] references with clickable links that scroll to 
                            the source section and highlight it briefly
                            """
                            if not text or not sources:
                                return text
                                
                            # Create session state for selected source if it doesn't exist
                            if "selected_source" not in st.session_state:
                                st.session_state.selected_source = None
                                
                            # Function to show source when clicked (no longer needed with anchor links)
                            def show_source(source_id):
                                st.session_state.selected_source = source_id
                                
                            # Find all [Source X] references
                            pattern = r'\[Source\s+(\d+)\]'
                            
                            # Split text by source references
                            parts = re.split(pattern, text)
                            
                            if len(parts) <= 1:  # No source references found
                                return text
                                
                            # Process parts and build the output with enhanced links
                            result = parts[0]
                            
                            # Add some custom CSS for better source highlighting when clicked
                            # We'll add the style and script to a dedicated section at the start of the analysis
                            # Add this to the session state if it hasn't been added yet
                            if "source_highlight_added" not in st.session_state:
                                highlight_style = """
                                <style>
                                .source-link {
                                    color: #4CAF50;
                                    text-decoration: none;
                                    font-weight: bold;
                                    padding: 2px 5px;
                                    border-radius: 4px;
                                    background-color: #f0f8ff;
                                    transition: background-color 0.3s;
                                }
                                .source-link:hover {
                                    background-color: #e0f0e0;
                                    text-decoration: underline;
                                }
                                @keyframes highlight-source {
                                    0% { background-color: #ffee99; }
                                    100% { background-color: transparent; }
                                }
                                .highlight-source-target {
                                    animation: highlight-source 3s ease-out;
                                }
                                </style>
                                
                                <script>
                                // Add click listeners to highlight the target source
                                document.addEventListener('DOMContentLoaded', function() {
                                    // Handle clicks on source links
                                    document.querySelectorAll('.source-link').forEach(link => {
                                        link.addEventListener('click', function() {
                                            // Get the target ID from the href
                                            const targetId = this.getAttribute('href');
                                            // Find the target element
                                            const targetElement = document.querySelector(targetId);
                                            if (targetElement) {
                                                // Add highlight class
                                                targetElement.classList.add('highlight-source-target');
                                                // Remove highlight class after animation completes
                                                setTimeout(() => {
                                                    targetElement.classList.remove('highlight-source-target');
                                                }, 3000);
                                            }
                                        });
                                    });
                                });
                                </script>
                                """
                                st.markdown(highlight_style, unsafe_allow_html=True)
                                st.session_state.source_highlight_added = True
                            
                            for i in range(1, len(parts)):
                                if i % 2 == 1:  # This is a source number
                                    source_num = parts[i]
                                    # Create enhanced link for source reference with custom class
                                    result += f'<a href="#source-heading-{source_num}" id="source-ref-{source_num}" class="source-link">[Source {source_num}]</a>'
                                else:  # This is text after a source reference
                                    result += parts[i]
                            
                            return result
                            
                        # Show sources in an expander
                        if sources and isinstance(sources, list) and len(sources) > 0:
                            # Removed Debug Source Data section as requested
                            
                            # Process the answer text to make source references clickable
                            if "answer_text" in locals():
                                processed_answer = process_source_references(answer_text, sources)
                                # Display the processed answer with clickable references
                                st.markdown(processed_answer, unsafe_allow_html=True)
                            
                            # Only show the "View Source" button if we have sources
                            if sources and isinstance(sources, list) and len(sources) > 0:
                                # Add a single "View Source" button at the bottom of the answer
                                st.markdown("---")
                                
                                # Create columns for better button layout
                                col1, col2, col3 = st.columns([1, 2, 1])
                                
                                with col2:
                                    # Use a key for the button to avoid rerun issues
                                    source_view_btn = st.button(
                                        "📄 View Source Document", 
                                        key="view_source_btn",
                                        use_container_width=True, 
                                        type="primary"
                                    )
                                    
                                    # Store button state in session to handle page refreshes
                                    if source_view_btn:
                                        st.session_state.show_pdf_source = True
                                        # Store available sources in session state
                                        if "pdf_source_pages" not in st.session_state:
                                            st.session_state.pdf_source_pages = []
                                        
                                        # Extract valid sources for PDF viewing
                                        valid_sources = []
                                        for i, source in enumerate(sources):
                                            if not isinstance(source, dict):
                                                continue
                                            
                                            # Get the collection ID and page number
                                            collection_id = source.get("collection_id", "Unknown")
                                            page_num = source.get("page", source.get("page_number", "Unknown"))
                                            
                                            if collection_id and collection_id != "Unknown" and page_num and page_num != "Unknown":
                                                valid_sources.append({
                                                    "index": i,
                                                    "collection_id": collection_id,
                                                    "page_num": page_num
                                                })
                                        
                                        # Store the valid sources in session state
                                        st.session_state.pdf_source_pages = valid_sources
                                        
                                        # Set the currently viewed source to the first one
                                        if valid_sources:
                                            if "current_pdf_source_index" not in st.session_state:
                                                st.session_state.current_pdf_source_index = 0
                                        
                                    # Check if we should show the source from session state
                                    if st.session_state.get('show_pdf_source', False):
                                        # Make sure we have at least one valid source
                                        if not st.session_state.get('pdf_source_pages', []):
                                            st.error("No valid PDF sources available.")
                                        else:
                                            # Get the current source to display
                                            current_idx = st.session_state.get('current_pdf_source_index', 0)
                                            if current_idx >= len(st.session_state.pdf_source_pages):
                                                current_idx = 0
                                                st.session_state.current_pdf_source_index = 0
                                                
                                            current_source = st.session_state.pdf_source_pages[current_idx]
                                            collection_id = current_source['collection_id']
                                            page_num = current_source['page_num']
                                            
                                            print(f"DEBUG: Attempting to load source: collection_id={collection_id}, page={page_num}")
                                            
                                            # Make sure we have the PDF processor
                                            if "pdf_processor" not in st.session_state:
                                                # Import the needed modules
                                                from pdf_processor import PDFProcessor, find_collections, load_pdf_metadata
                                                
                                                # Initialize PDF processor with current model settings
                                                print(f"DEBUG: Creating PDF processor with model: {st.session_state.model_provider}, {st.session_state.get('current_model')}")
                                                pdf_processor = PDFProcessor(st.session_state.model_provider, st.session_state.get('current_model'))
                                                st.session_state.pdf_processor = pdf_processor
                                            
                                            # Ensure we have PDF collections data loaded
                                            if 'pdf_collections' not in st.session_state or not st.session_state.pdf_collections:
                                                st.session_state.pdf_collections = {}
                                                pdf_collections = find_collections()
                                                print(f"DEBUG: Available PDF collections: {pdf_collections}")
                                                
                                                for cid in pdf_collections:
                                                    try:
                                                        metadata = st.session_state.pdf_processor.get_pdf_metadata(cid)
                                                        if metadata:
                                                            st.session_state.pdf_collections[cid] = metadata
                                                            print(f"DEBUG: Loaded metadata for {cid}: {metadata.get('filename')}")
                                                    except Exception as e:
                                                        print(f"DEBUG: Error loading metadata for {cid}: {str(e)}")
                                            
                                            # Extract query terms for highlighting
                                            query_terms = []
                                            if "query" in st.session_state:
                                                # Use the query to extract terms for highlighting
                                                from pdf_processor import extract_query_terms
                                                query_terms = extract_query_terms(st.session_state.query)
                                                print(f"DEBUG: Extracted query terms for highlighting: {query_terms}")
                                            
                                            # Instead of causing a rerun/refresh, we'll display the content directly here
                                            st.markdown("---")
                                            
                                            # Show source navigation header with pagination
                                            total_sources = len(st.session_state.pdf_source_pages)
                                            st.subheader(f"Source Document - Page {page_num} (Source {current_idx + 1} of {total_sources})")
                                            
                                            # Source navigation buttons
                                            if total_sources > 1:
                                                col1, col2, col3 = st.columns([1, 2, 1])
                                                with col1:
                                                    if current_idx > 0:
                                                        if st.button("← Previous Source", key="prev_source_btn"):
                                                            st.session_state.current_pdf_source_index = current_idx - 1
                                                            st.rerun()
                                                
                                                with col3:
                                                    if current_idx < total_sources - 1:
                                                        if st.button("Next Source →", key="next_source_btn"):
                                                            st.session_state.current_pdf_source_index = current_idx + 1
                                                            st.rerun()
                                            
                                            # Get metadata for the collection
                                            metadata = {}
                                            if "pdf_collections" in st.session_state and collection_id in st.session_state.pdf_collections:
                                                metadata = st.session_state.pdf_collections[collection_id]
                                                print(f"Using cached metadata for {collection_id}")
                                            else:
                                                # Try to load metadata directly
                                                try:
                                                    metadata = st.session_state.pdf_processor.get_pdf_metadata(collection_id)
                                                    if metadata:
                                                        # Also cache it for future use
                                                        st.session_state.pdf_collections[collection_id] = metadata
                                                        print(f"Loaded and cached new metadata for {collection_id}")
                                                except Exception as e:
                                                    st.error(f"Error loading metadata: {str(e)}")
                                            
                                            # Display the PDF viewer
                                            try:
                                                # Before anything, check what PDFs are actually available
                                                import glob
                                                available_pdfs = glob.glob("uploaded_pdfs/*.pdf")
                                                print(f"DEBUG: Actually available PDFs in uploaded_pdfs: {available_pdfs}")
                                                
                                                # Check for PDFs that match the collection ID (most reliable)
                                                matching_pdfs = [pdf for pdf in available_pdfs if collection_id in pdf]
                                                if matching_pdfs:
                                                    print(f"DEBUG: Found PDF matching collection ID {collection_id}: {matching_pdfs[0]}")
                                                    if metadata is None:
                                                        metadata = {}
                                                    metadata["path"] = matching_pdfs[0]
                                                    
                                                    # Also update cached metadata
                                                    if "pdf_collections" in st.session_state:
                                                        if collection_id not in st.session_state.pdf_collections:
                                                            st.session_state.pdf_collections[collection_id] = {}
                                                        st.session_state.pdf_collections[collection_id]["path"] = matching_pdfs[0]
                                                    
                                                    # Save to disk for future reference
                                                    if hasattr(st.session_state.pdf_processor, 'pdf_metadata'):
                                                        if collection_id not in st.session_state.pdf_processor.pdf_metadata:
                                                            st.session_state.pdf_processor.pdf_metadata[collection_id] = {}
                                                        st.session_state.pdf_processor.pdf_metadata[collection_id]["path"] = matching_pdfs[0]
                                                        from pdf_processor import save_pdf_metadata
                                                        save_pdf_metadata(st.session_state.pdf_processor.pdf_metadata)
                                                        print(f"DEBUG: Updated saved metadata on disk for {collection_id}")
                                                
                                                # Check if the file exists and print the path for debugging
                                                if metadata and "path" in metadata:
                                                    pdf_path = metadata.get("path")
                                                    print(f"DEBUG: Checking PDF file path: {pdf_path}")
                                                        
                                                    if os.path.exists(pdf_path):
                                                        print(f"DEBUG: PDF file exists at {pdf_path}")
                                                    else:
                                                        print(f"DEBUG: PDF file NOT found at {pdf_path}, trying alternate paths")
                                                            
                                                        # Try possible locations with additional fallbacks
                                                        possible_paths = [
                                                            # Try exact matches with available PDFs first
                                                            *matching_pdfs,
                                                            # Common relative paths
                                                            os.path.join("uploaded_pdfs", f"{collection_id}.pdf"),
                                                            os.path.join(".", "uploaded_pdfs", f"{collection_id}.pdf"),
                                                            os.path.join("uploaded_pdfs", f"{collection_id.replace('pdf_', '')}.pdf"),
                                                            # Try with original filename if available
                                                            os.path.join("uploaded_pdfs", metadata.get("filename", "")),
                                                            # Try any available PDF as fallback (if we've got exactly 1)
                                                            available_pdfs[0] if len(available_pdfs) == 1 else None,
                                                        ]
                                                            
                                                        # Filter out None or empty paths and remove duplicates
                                                        possible_paths = [p for p in possible_paths if p]
                                                        possible_paths = list(dict.fromkeys(possible_paths))  # Remove duplicates
                                                        print(f"DEBUG: Trying alternate paths: {possible_paths}")
                                                            
                                                        for alt_path in possible_paths:
                                                            if os.path.exists(alt_path):
                                                                print(f"DEBUG: PDF file found at alternative path: {alt_path}")
                                                                # Update metadata with the correct path
                                                                metadata["path"] = alt_path
                                                                # Also update cached metadata
                                                                if "pdf_collections" in st.session_state and collection_id in st.session_state.pdf_collections:
                                                                    st.session_state.pdf_collections[collection_id]["path"] = alt_path
                                                                    print(f"DEBUG: Updated cached metadata path for {collection_id}")
                                                                # Save to disk for future reference
                                                                if hasattr(st.session_state.pdf_processor, 'pdf_metadata'):
                                                                    if collection_id in st.session_state.pdf_processor.pdf_metadata:
                                                                        st.session_state.pdf_processor.pdf_metadata[collection_id]["path"] = alt_path
                                                                        from pdf_processor import save_pdf_metadata
                                                                        save_pdf_metadata(st.session_state.pdf_processor.pdf_metadata)
                                                                        print(f"DEBUG: Updated saved metadata on disk for {collection_id}")
                                                                break
                                                
                                                pdf_viewer_html = st.session_state.pdf_processor.create_interactive_pdf_viewer(
                                                    collection_id=collection_id,
                                                    metadata=metadata,
                                                    pdf_processor=st.session_state.pdf_processor,
                                                    query_terms=query_terms,
                                                    highlight_page=int(page_num)
                                                )
                                                
                                                # Check for error messages in the HTML
                                                if pdf_viewer_html.startswith("<p>Error") or pdf_viewer_html.startswith("<p>PDF not available"):
                                                    st.warning("Could not display the original PDF. Showing text content instead.")
                                                    st.markdown(pdf_viewer_html, unsafe_allow_html=True)
                                                else:
                                                    st.markdown(pdf_viewer_html, unsafe_allow_html=True)
                                                    
                                            except Exception as e:
                                                st.error(f"Error displaying PDF source: {str(e)}")
                                            
                                            # Add a close button to hide the PDF viewer
                                            if st.button("Close PDF View", key="close_pdf_btn"):
                                                st.session_state.show_pdf_source = False
                                                st.rerun()
                                
                                # Add a small debug expander for developers
                                with st.expander("Source Details", expanded=False):
                                    st.write(f"Number of sources: {len(sources)}")
                                    for i, source in enumerate(sources):
                                        if not isinstance(source, dict):
                                            continue
                                            
                                        # Get source details
                                        collection_id = source.get("collection_id", "Unknown")
                                        page_num = source.get("page", source.get("page_number", "Unknown"))
                                        
                                        # Get the file name from metadata
                                        file_name = "Unknown"
                                        if collection_id in st.session_state.pdf_collections:
                                            file_name = st.session_state.pdf_collections[collection_id].get("filename", "Unknown")
                                        
                                        st.write(f"Source {i+1}: {file_name}, Page {page_num}")
                                        
                                        # Show content snippets if available
                                        content = ""
                                        if "text" in source and isinstance(source["text"], str):
                                            content = source["text"]
                                        elif "content" in source and isinstance(source["content"], str):
                                            content = source["content"]
                                        elif "page_content" in source and isinstance(source["page_content"], str):
                                            content = source["page_content"]
                                        
                                        if content:
                                            st.text_area(f"Content from source {i+1}", value=content[:500] + "..." if len(content) > 500 else content, height=100)
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
    
    elif st.session_state.data_source_mode == "image":
        # Image Analysis interface
        from image_analyzer import image_analyzer_ui
        
        # Display the image analyzer UI
        image_analyzer_ui()

# For database-related modes, we need a connection
elif st.session_state.connected:
    # Check if we need to show tables, reports, or web content mode
    if st.session_state.data_source_mode == "tables":
        # Tables must be selected in tables mode
        if not st.session_state.selected_tables:
            st.info("Please select at least one table from the sidebar to start querying")
    elif st.session_state.data_source_mode == "reports":
        # Reports must be selected in reports mode
        if not st.session_state.selected_reports:
            st.info("Please select at least one report from the sidebar to start querying")
        else:
            # Create a temporary view for the selected report
            try:
                # We'll just take the first selected report for now
                report_name = st.session_state.selected_reports[0]
                
                # Execute the report and create a view
                _, error, view_name = reports.execute_report(st.session_state.conn, report_name, create_view=True)
                
                if error:
                    st.error(error)
                elif view_name:
                    st.session_state.report_view_name = view_name
                    
                    # Get schema for the view
                    view_schema_query = f"SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = '{view_name}'"
                    try:
                        view_schema = database.execute_sql_query(st.session_state.conn, view_schema_query)
                        
                        # Create a table info structure similar to what we use for database tables
                        if not view_schema.empty:
                            # Transform the view schema to match the schema format expected by the NL->SQL function
                            # This ensures compatibility with the schema format used by regular tables
                            formatted_schema = pd.DataFrame({
                                "column_name": view_schema["column_name"],
                                "data_type": view_schema["data_type"],
                                "is_nullable": view_schema["is_nullable"]
                            })
                            
                            st.session_state.tables_info[view_name] = formatted_schema
                            
                            # Add the view to selected tables so the NL->SQL knows about it
                            if 'selected_tables_for_report' not in st.session_state:
                                st.session_state.selected_tables_for_report = []
                            
                            st.session_state.selected_tables_for_report = [view_name]
                    except Exception as e:
                        st.warning(f"Could not get schema for view: {str(e)}")
            except Exception as e:
                st.error(f"Error preparing report data: {str(e)}")
    
    # Only continue if tables or reports are selected for database modes
    if ((st.session_state.data_source_mode == "tables" and st.session_state.selected_tables) or 
        (st.session_state.data_source_mode == "reports" and st.session_state.selected_reports)):
        
        # Database query interface (tables or reports)
        st.header("Ask Questions About Your Data")
        
        query = st.text_area(
            "Enter your question in natural language",
            value="",  # Don't initialize with previous query
            key="db_query_input",
            height=100,
            placeholder="Example: Show me the total sales by region for the last quarter",
            help="Ask questions about your data in plain English."
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            execute = st.button("Submit", use_container_width=True, type="primary")
        
        with col2:
            clear = st.button("Clear Results", use_container_width=True)
            
        if clear:
            st.session_state.current_query = ""
            st.session_state.current_sql = ""
            st.session_state.current_results = pd.DataFrame()
            st.rerun()
            
        if execute and query:
            st.session_state.current_query = query
            
            with st.spinner("Processing your question..."):
                # Database modes only (tables or reports)
                if st.session_state.data_source_mode == "tables":
                    # Tables mode: Get schema for selected tables
                    selected_tables_info = {table: st.session_state.tables_info[table] 
                                          for table in st.session_state.selected_tables}
                    
                    # Get database type from session state
                    db_type = st.session_state.get('db_type', 'postgresql')
                    
                    # Use chosen agent workflow
                    if st.session_state.use_multi_agent:
                        if st.session_state.use_enhanced_multi_agent:
                            # Get specialized SQL model settings from session state
                            use_specialized_sql_model = st.session_state.get('use_specialized_sql_model', False)
                            specialized_sql_model = st.session_state.get('specialized_sql_model', "gpt-4o")
                            
                            # Use enhanced multi-agent workflow with additional agents
                            agent_results = agents.enhanced_multi_agent_workflow(
                                question=query,
                                tables_info=selected_tables_info,
                                conn=st.session_state.conn,
                                db_type=db_type,
                                use_specialized_sql_model=use_specialized_sql_model,
                                specialized_sql_model=specialized_sql_model,
                                model_provider=st.session_state.model_provider,
                                model=st.session_state.get('current_model')
                            )
                        else:
                            # Use standard multi-agent workflow
                            agent_results = agents.multi_agent_workflow(
                                question=query,
                                tables_info=selected_tables_info,
                                conn=st.session_state.conn,
                                db_type=db_type,
                                model_provider=st.session_state.model_provider,
                                model=st.session_state.get('current_model')
                            )
                            
                        # Get the SQL query from the agent results
                        sql_query = agent_results.get('sql_query')
                        
                        # Store the agent results for later use in visualization and explanation
                        st.session_state.agent_results = agent_results
                    else:
                        # Use the standard approach for NL to SQL translation
                        sql_query = nlp.natural_language_to_sql(query, selected_tables_info, st.session_state.selected_tables, db_type)
                        # Clear any previous agent results
                        if 'agent_results' in st.session_state:
                            del st.session_state.agent_results
                else:
                    # Reports mode: Use temporary view created from report
                    if 'report_view_name' in st.session_state and st.session_state.report_view_name:
                        # Get the view name for the temporary view created from the report
                        view_name = st.session_state.report_view_name
                        
                        # Get table info for the view
                        if view_name in st.session_state.tables_info:
                            view_info = {view_name: st.session_state.tables_info[view_name]}
                            
                            # Use NL to SQL with the view as if it was a table
                            try:
                                # Get database type from session state
                                db_type = st.session_state.get('db_type', 'postgresql')
                                
                                # We'll pass the view name as the only "table" to be used in SQL generation
                                sql_query = nlp.natural_language_to_sql(query, view_info, [view_name], db_type)
                            except Exception as e:
                                st.error(f"Error generating SQL for report view: {str(e)}")
                                sql_query = None
                        else:
                            st.error(f"View schema information is not available for {view_name}")
                            sql_query = None
                    else:
                        # Fallback if view creation failed: use original report SQL
                        try:
                            # Get the report SQL - for now we'll just use the first selected report
                            if st.session_state.selected_reports:
                                selected_report = st.session_state.selected_reports[0]
                                sql_query = reports.get_report_query(selected_report)
                            else:
                                sql_query = None
                        except Exception as e:
                            st.error(f"Error processing report query: {str(e)}")
                            sql_query = None
                
                if sql_query:
                    st.session_state.current_sql = sql_query
                    
                    # Put the technical details in an expander
                    with st.expander("Query Details", expanded=False):
                        # Format and display SQL
                        st.subheader("Generated SQL")
                        formatted_sql = utils.format_sql(sql_query)
                        st.code(formatted_sql, language="sql")
                        
                        # Display the explanation
                        explanation = nlp.explain_query(sql_query)
                        st.subheader("Query Explanation")
                        st.write(explanation)
                    
                    # Execute query
                    with st.spinner("Retrieving data..."):
                        start_time = time.time()
                        try:
                            # Check if the SQL query is valid
                            if not sql_query or len(sql_query.strip()) == 0:
                                st.error("Generated SQL query is empty. Please try rephrasing your question.")
                                st.session_state.current_results = pd.DataFrame()
                                query_time = utils.calculate_query_time(start_time)
                            else:
                                # Execute the query differently based on mode
                                if st.session_state.data_source_mode == "tables":
                                    # Regular SQL execution for tables mode
                                    results = database.execute_sql_query(st.session_state.conn, sql_query)
                                    st.session_state.current_results = results
                                elif st.session_state.data_source_mode == "reports":
                                    # For reports mode, we have two options:
                                    # 1. If we have a temporary view, query it using the generated SQL
                                    # 2. If not, fall back to the original report query
                                    if 'report_view_name' in st.session_state and st.session_state.report_view_name:
                                        # We have a temporary view, so execute the generated SQL against it
                                        try:
                                            results = database.execute_sql_query(st.session_state.conn, sql_query)
                                            st.session_state.current_results = results
                                        except Exception as e:
                                            # If querying the view fails, fall back to the original report
                                            st.warning(f"Error querying view: {str(e)}. Falling back to original report.")
                                            selected_report = st.session_state.selected_reports[0]
                                            result = reports.execute_report(st.session_state.conn, selected_report)
                                            # Handle return values based on length of tuple
                                            if len(result) == 3:
                                                results, error_message, _ = result
                                            else:
                                                results, error_message = result
                                            
                                            if error_message:
                                                raise Exception(error_message)
                                            
                                            st.session_state.current_results = results
                                    else:
                                        # No temporary view available, use the original report
                                        selected_report = st.session_state.selected_reports[0]
                                        result = reports.execute_report(st.session_state.conn, selected_report)
                                        # Handle return values based on length of tuple
                                        if len(result) == 3:
                                            results, error_message, _ = result
                                        else:
                                            results, error_message = result
                                        
                                        if error_message:
                                            raise Exception(error_message)
                                        
                                        st.session_state.current_results = results
                                elif st.session_state.data_source_mode == "web":
                                    # Web content mode - should not reach here as web content is handled directly
                                    # by the web_scraper module, but include as a fallback
                                    st.warning("Direct SQL execution is not supported for web content. Please use the web content interface.")
                                    st.session_state.current_results = pd.DataFrame()
                                else:
                                    # Unexpected mode
                                    st.error(f"Unsupported data source mode: {st.session_state.data_source_mode}")
                                    st.session_state.current_results = pd.DataFrame()
                                
                                query_time = utils.calculate_query_time(start_time)
                        except Exception as e:
                            error_msg = str(e)
                            
                            # Create more user-friendly error message
                            error_container = st.error("There was a problem with your query.")
                            
                            with st.expander("Error details", expanded=True):
                                st.error(f"Error: {error_msg}")
                                
                                # Provide user-friendly suggestions based on error types
                                if "relation" in error_msg and "does not exist" in error_msg:
                                    st.info("💡 It looks like the system tried to query a table that doesn't exist. Try selecting different tables from the sidebar or rephrasing your question to reference only the available tables.")
                                elif "column" in error_msg and "does not exist" in error_msg:
                                    st.info("💡 The query referenced a column that doesn't exist in your tables. Try rephrasing your question using column names that match your database schema.")
                                elif "syntax error" in error_msg:
                                    st.info("💡 There was a syntax error in the generated SQL query. Please try rephrasing your question to be more specific about what you're looking for.")
                                elif "permission denied" in error_msg:
                                    st.info("💡 You don't have permission to access some of the requested data. Try selecting different tables or asking about data you have access to.")
                                else:
                                    st.info("💡 Please try rephrasing your question to be clearer or more specific. Focus on the tables and columns you've selected in the sidebar.")
                            
                            st.session_state.current_results = pd.DataFrame()
                            query_time = utils.calculate_query_time(start_time)
                        
                        # Add to query history
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        if query not in [q[0] for q in st.session_state.query_history]:
                            st.session_state.query_history.append((query, sql_query, timestamp))
                        
                        # Keep only the last 10 queries
                        if len(st.session_state.query_history) > 10:
                            st.session_state.query_history = st.session_state.query_history[-10:]
                    
                    # Display query time
                    st.caption(f"Query executed in {query_time}")
                elif st.session_state.data_source_mode != "web":
                    # Only show SQL-related error messages for database and reports modes (not web content)
                    st.error("Failed to generate a SQL query.")
                    
                    with st.expander("Troubleshooting tips", expanded=True):
                        st.markdown("""
                        ### 💡 Tips for better results:
                        
                        1. **Be more specific** - Clearly state what information you're looking for
                        2. **Reference tables by name** - Mention specific tables from those you've selected
                        3. **Simplify your question** - Break complex questions into simpler parts
                        4. **Check table selection** - Make sure you've selected all relevant tables
                        5. **Use proper context** - Include time periods or categories if applicable
                        
                        ### Example queries:
                        - "Show me total sales by product category for last month"
                        - "What are the top 10 customers by order value?"
                        - "Count unique users who placed orders in January 2023"
                        """)
                
        # Display results if available
        if not st.session_state.current_results.empty:
            st.header("Query Results")
            
            # Get query metadata based on mode
            if st.session_state.data_source_mode == "tables":
                # Tables mode
                # Extract metadata for visualization, providing more context for the NLP function
                try:
                    metadata = nlp.extract_query_metadata(st.session_state.current_query, 
                                                        st.session_state.current_sql, 
                                                        st.session_state.tables_info)
                except Exception as e:
                    st.warning(f"Could not extract detailed query metadata: {str(e)}")
                    # Fallback to simpler metadata
                    metadata = {
                        "query_type": "simple",
                        "visualization_type": "table",
                        "columns": list(st.session_state.current_results.columns),
                        "visualization_types": ["table", "bar", "line"] if not st.session_state.current_results.empty else []
                    }
            elif st.session_state.data_source_mode == "reports":
                # Reports mode - we need to adapt the metadata structure
                # For reports, we'll use a similar approach to tables to ensure consistent visualization behavior
                try:
                    # Get column info from the report
                    report_name = st.session_state.selected_reports[0] if st.session_state.selected_reports else "Unknown"
                    
                    # Using similar metadata format as tables mode but with report-specific info
                    metadata = {
                        "query_type": "simple",  # Changed from "report" to "simple" to avoid early returns in visualization.py
                        "visualization_type": "bar",  # Default visualization type
                        "report_name": report_name,
                        "columns": list(st.session_state.current_results.columns)
                    }
                    
                    # If we have a natural language query for the report, try to extract better metadata
                    if st.session_state.current_query:
                        try:
                            # Extract metadata from the natural language query
                            report_metadata = nlp.extract_query_metadata(
                                st.session_state.current_query,
                                st.session_state.current_sql,
                                {"report_view": st.session_state.current_results}  # Use results as table info
                            )
                            # Merge report-specific metadata with the extracted metadata
                            metadata.update(report_metadata)
                        except Exception as e:
                            # Keep using the default metadata if extraction fails
                            pass
                except Exception as e:
                    # Fallback to basic metadata
                    metadata = {
                        "query_type": "simple",
                        "visualization_type": "table",
                        "columns": list(st.session_state.current_results.columns)
                    }
            else:
                # Web Content mode - simple metadata for visualization
                metadata = {
                    "query_type": "web_search",
                    "visualization_type": "table",
                    "columns": list(st.session_state.current_results.columns),
                    "visualization_types": ["table"] if not st.session_state.current_results.empty else [],
                    "source": st.session_state.selected_collection if st.session_state.selected_collection else "web"
                }
            
            # Detect column types
            column_types = visualization.detect_column_types(st.session_state.current_results)
            
            # Display result summary
            summary = utils.get_dataframe_summary(st.session_state.current_results)
            st.caption(f"Results: {summary['rows']} rows, {summary['columns']} columns")
            
            # Display tabular results with error handling for null/undefined values
            try:
                # Display a message showing we're rendering the results safely
                # st.info("Rendering results safely to avoid JavaScript errors...")
                
                # Special handling for COUNT queries or single value results (common cause of JS errors)
                is_singleton_result = (st.session_state.current_results.shape[0] == 1 and 
                                      st.session_state.current_results.shape[1] == 1)
                
                # If this is a single value result (like a COUNT query), display differently
                if is_singleton_result:
                    # Get the column name and value
                    col_name = st.session_state.current_results.columns[0]
                    value = st.session_state.current_results.iloc[0, 0]
                    
                    # Handle special case for COUNT(*) column name, which often causes issues
                    if 'count(' in col_name.lower():
                        # Rename the column to something more user-friendly
                        col_name = "Row Count"
                        # Also update the DataFrame column name
                        st.session_state.current_results.columns = [col_name]
                    
                    # Check if the column name contains 'count' or similar keywords
                    count_keywords = ['count', 'sum', 'total', 'amount', 'number']
                    is_count_col = any(keyword in col_name.lower() for keyword in count_keywords)
                    
                    # Create a better display for count/single value results
                    st.subheader("Result:")
                    st.markdown(f"### {col_name}: **{value}**")
                    
                    # Add a basic visualization for count results
                    if is_count_col:
                        try:
                            # Create a gauge-like visualization
                            fig = go.Figure(go.Indicator(
                                mode="number",
                                value=float(value) if pd.api.types.is_numeric_dtype(type(value)) else 0,
                                title={"text": col_name},
                                domain={'x': [0, 1], 'y': [0, 1]}
                            ))
                            fig.update_layout(height=250)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # For count queries, create a simple bar chart as well
                            bar_fig = go.Figure(data=[
                                go.Bar(
                                    x=[col_name],
                                    y=[float(value) if pd.api.types.is_numeric_dtype(type(value)) else 0],
                                    text=[value],
                                    textposition='auto',
                                    width=[0.5]
                                )
                            ])
                            bar_fig.update_layout(
                                title=f"{col_name} Visualization",
                                height=350,
                                yaxis=dict(title=col_name)
                            )
                            st.plotly_chart(bar_fig, use_container_width=True)
                        except Exception as viz_e:
                            # Don't worry if this fails; the text display is sufficient
                            pass
                    
                    # Also display in an HTML table which is more robust than dataframe
                    try:
                        html = f"""
                        <table style="width:100%; border-collapse: collapse;">
                          <thead>
                            <tr style="background-color: #f2f2f2;">
                              <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">{col_name}</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td style="padding: 12px; text-align: left; border: 1px solid #ddd;">{value}</td>
                            </tr>
                          </tbody>
                        </table>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                    except Exception as e:
                        st.info(f"Result value: {value}")
                    
                    # Skip the rest of the visualization code
                    st.caption("Note: Advanced visualizations are not generated for single-value results.")
                    st.stop()
                else:
                    # First try showing the data using a normal dataframe display
                    # But catch any errors that occur
                    try:
                        # Clean the data first
                        safe_df = pd.DataFrame()
                        display_df = st.session_state.current_results.copy()
                        
                        # Convert all data to safer types to avoid JavaScript errors
                        for col in display_df.columns:
                            try:
                                # Handle different column types appropriately
                                if pd.api.types.is_numeric_dtype(display_df[col]):
                                    # Convert numeric columns and fill NaN with 0
                                    safe_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0)
                                elif pd.api.types.is_datetime64_dtype(display_df[col]) or pd.api.types.is_datetime64_any_dtype(display_df[col]):
                                    # Convert datetime columns to strings
                                    safe_df[col] = pd.to_datetime(display_df[col], errors='coerce').fillna(pd.Timestamp('now')).dt.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    # Everything else as strings
                                    safe_df[col] = display_df[col].fillna("").astype(str)
                            except Exception as e:
                                # If conversion fails, use string
                                safe_df[col] = display_df[col].fillna("").astype(str)
                        
                        # Try displaying the cleaned dataframe
                        st.dataframe(safe_df, use_container_width=True)
                    except Exception as table_error:
                        # If the table display fails, show fallback info
                        st.error(f"Could not display data table due to error: {str(table_error)}")
                        st.info(f"""
                        **Query results contain {st.session_state.current_results.shape[0]} rows and {st.session_state.current_results.shape[1]} columns.**
                        
                        The data table view has been disabled because of an error. 
                        You can:
                        - Use the visualizations below to explore your data
                        - Download the complete results using the CSV button
                        - Try a different query if you need to see specific records
                        """)
                        
                        # Show column names for reference when table fails
                        st.caption("**Columns in result:** " + ", ".join(st.session_state.current_results.columns))
                    
            except Exception as e:
                st.error(f"Error displaying results: {str(e)}")
                
                # Ultimate fallback - show the results as text
                st.warning("Displaying results as text due to formatting issues.")
                st.text(str(st.session_state.current_results))
            
            # Download link
            csv, filename = utils.create_download_link(
                st.session_state.current_results, 
                f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
            # Visualizations
            st.header("Visualizations")
            
            # Display enhanced analytics if multi-agent mode was used
            if st.session_state.use_multi_agent and 'agent_results' in st.session_state:
                # Get the data analysis from the multi-agent results
                analysis_results = st.session_state.agent_results.get('analysis', {})
                
                # Display the enhanced analysis
                st.subheader("Enhanced Data Analysis")
                
                # Display main insights
                if 'insights' in analysis_results:
                    st.markdown("### Key Insights")
                    for insight in analysis_results['insights']:
                        st.markdown(f"- {insight}")
                
                # Display explanation
                # Display explanation - now with streaming support
                if 'explanation_streaming' in analysis_results:
                    with st.expander("Detailed Analysis", expanded=True):
                        # Create a placeholder for streaming output
                        explanation_placeholder = st.empty()
                        
                        # Call the streaming function with our placeholder
                        analysis_results['explanation_streaming'](explanation_placeholder)
                elif 'explanation' in analysis_results:
                    # Fallback to non-streaming explanation if streaming is not available
                    with st.expander("Detailed Analysis", expanded=True):
                        st.markdown(analysis_results['explanation'])
                
                # Automatically generate a visualization based on agent recommendations
                if 'visualization_recommendations' in analysis_results:
                    viz_rec = analysis_results['visualization_recommendations']
                    st.subheader("AI-Generated Visualization")
                    
                    try:
                        if 'viz_type' in viz_rec and 'columns' in viz_rec:
                            # Log visualization recommendations for debugging
                            st.session_state['viz_debug_info'] = {
                                'viz_type': viz_rec['viz_type'],
                                'columns': viz_rec['columns'],
                                'title': viz_rec.get('title', 'AI-Generated Visualization'),
                                'full_recommendation': viz_rec,
                                'full_analysis': {k: v for k, v in analysis_results.items() if k != 'explanation'}
                            }
                            
                            # First, check if the AI provided Python code for the visualization
                            if 'visualization_code' in viz_rec and viz_rec['visualization_code']:
                                try:
                                    # Display tabs for visualization and code
                                    viz_tabs = st.tabs(["Visualization", "Python Code"])
                                    
                                    with viz_tabs[0]:
                                        # Extract the Python code
                                        viz_code = viz_rec['visualization_code']
                                        
                                        # Clean up the code if it's in a markdown code block
                                        if "```python" in viz_code:
                                            viz_code = viz_code.split("```python")[1]
                                            if "```" in viz_code:
                                                viz_code = viz_code.split("```")[0]
                                        elif "```" in viz_code:
                                            viz_code = viz_code.split("```")[1]
                                            if "```" in viz_code:
                                                viz_code = viz_code.split("```")[0]
                                        
                                        # Create a local namespace for execution
                                        local_vars = {
                                            'df': st.session_state.current_results,
                                            'pd': pd,
                                            'np': np,
                                            'px': px,
                                            'go': go,
                                            'make_subplots': make_subplots
                                        }
                                        
                                        # Execute the code
                                        if st.session_state.get('debug_mode', True):
                                            st.write("Executing AI-generated visualization code...")
                                        
                                        # Add safety to the code execution
                                        viz_code = f"""
# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

{viz_code.strip()}

# Call the function with our data
fig = create_visualization(df)
"""
                                        
                                        # Execute the code
                                        exec(viz_code, globals(), local_vars)
                                        
                                        # Get the figure from the local namespace
                                        if 'fig' in local_vars and local_vars['fig'] is not None:
                                            # Display the visualization
                                            st.plotly_chart(local_vars['fig'], use_container_width=True)
                                            
                                            # Show explanation of why this visualization was chosen
                                            if 'description' in viz_rec:
                                                st.info(f"**Why this visualization was chosen:** {viz_rec['description']}")
                                        else:
                                            st.warning("The code executed but did not produce a figure.")
                                    
                                    with viz_tabs[1]:
                                        # Display the Python code
                                        st.code(viz_code, language="python")
                                        
                                        # Add a button to copy the code
                                        st.markdown("**Copy this code to use in your own analysis!**")
                                
                                except Exception as code_error:
                                    st.warning(f"Error executing AI-generated visualization code: {str(code_error)}")
                                    st.code(viz_rec['visualization_code'], language="python")
                                    
                                    # Fall back to the standard visualization method
                                    st.write("Falling back to standard visualization method...")
                                    use_standard_viz = True
                            else:
                                # If no code provided, use the standard method
                                use_standard_viz = True
                            
                            # Use standard visualization method if needed
                            if not 'visualization_code' in viz_rec or locals().get('use_standard_viz', False):
                                # Generate the recommended visualization
                                auto_viz_settings = {
                                    'title': viz_rec.get('title', 'AI-Generated Visualization'),
                                    'color_theme': 'vibrant',
                                    'height': 500,
                                    'show_legend': True
                                }
                                
                                # Ensure columns is properly formatted as a dictionary
                                columns = viz_rec['columns']
                                
                                # Debug info
                                if st.session_state.get('debug_mode', True):
                                    st.write(f"Column format from AI: {type(columns)}")
                                    st.write(f"Original columns value: {columns}")
                                
                                if not isinstance(columns, dict):
                                    if isinstance(columns, str):
                                        # Try to parse if it's a JSON string
                                        try:
                                            import json
                                            # Replace single quotes with double quotes if needed
                                            if "'" in columns and '"' not in columns:
                                                columns = columns.replace("'", '"')
                                            columns = json.loads(columns)
                                            if st.session_state.get('debug_mode', True):
                                                st.write(f"Parsed JSON string to dict: {columns}")
                                        except Exception as e:
                                            if st.session_state.get('debug_mode', True):
                                                st.write(f"Error parsing JSON: {str(e)}")
                                            # Create a simple default
                                            columns = {"x": columns}
                                    else:
                                        # Handle other non-dict formats safely
                                        if st.session_state.get('debug_mode', True):
                                            st.write(f"Could not convert columns to dict: {type(columns)}")
                                        columns = {}
                                
                                # Debug info before creating visualization
                                st.session_state['viz_debug_info']['df_info'] = {
                                    'shape': st.session_state.current_results.shape,
                                    'columns': st.session_state.current_results.columns.tolist(),
                                    'sample': st.session_state.current_results.head(2).to_dict()
                                }
                                
                                # Create the visualization using the AI's recommendations  
                                try:
                                    auto_fig = visualization.create_advanced_visualization(
                                        st.session_state.current_results,
                                        viz_rec['viz_type'],
                                        columns,
                                        auto_viz_settings
                                    )
                                    
                                    if auto_fig:
                                        # Display the AI-generated visualization
                                        st.plotly_chart(auto_fig, use_container_width=True)
                                        
                                        # Show explanation of why this visualization was chosen
                                        if 'description' in viz_rec:
                                            st.info(f"**Why this visualization was chosen:** {viz_rec['description']}")
                                    else:
                                        st.warning("Could not generate the recommended visualization.")
                                        # Show debug expander
                                        with st.expander("Debug Information", expanded=True):
                                            st.write("Visualization details:")
                                            st.json(st.session_state['viz_debug_info'])
                                except Exception as viz_error:
                                    st.warning(f"Error in visualization generation: {str(viz_error)}")
                                    # Show detailed debug information
                                    with st.expander("Debug Information", expanded=True):
                                        st.write("Visualization error details:")
                                        st.json(st.session_state['viz_debug_info'])
                                        st.write(f"Error: {str(viz_error)}")
                    except Exception as e:
                        st.warning(f"Error generating recommended visualization: {str(e)}")
                        # Show debug info in an expander
                        with st.expander("Debug Information", expanded=False):
                            st.write("Visualization error details:")
                            if 'viz_debug_info' in st.session_state:
                                st.json(st.session_state['viz_debug_info'])
                            st.write(f"Error: {str(e)}")
                
                # Enhanced Multi-Agent: Show specialized agent results
                if st.session_state.use_enhanced_multi_agent:
                    # Show validation info if available
                    validation_results = st.session_state.agent_results.get('sql_validation', {})
                    if validation_results:
                        with st.expander("SQL Validation Details", expanded=False):
                            st.markdown("### SQL Validation")
                            
                            # Show if any fixes were applied
                            if 'was_fixed' in validation_results:
                                if validation_results['was_fixed']:
                                    st.success("✅ The SQL query was automatically fixed")
                                    
                                    if 'fixes_applied' in validation_results:
                                        st.markdown("### Fixes Applied:")
                                        for fix in validation_results['fixes_applied']:
                                            st.markdown(f"- {fix}")
                                else:
                                    st.info("✓ The SQL query was validated and no issues were found")
                            
                            # Show validation details
                            if 'validation_details' in validation_results:
                                st.markdown("### Validation Details:")
                                st.markdown(validation_results['validation_details'])
                    
                    # Show testing info if available
                    testing_results = st.session_state.agent_results.get('query_testing', {})
                    if testing_results:
                        with st.expander("Query Testing Details", expanded=False):
                            st.markdown("### Query Testing")
                            
                            # Show test results
                            if 'test_passed' in testing_results:
                                if testing_results['test_passed']:
                                    st.success("✅ Query passed all tests")
                                else:
                                    st.warning("⚠️ Query testing found potential issues")
                            
                            # Show test details
                            if 'test_details' in testing_results:
                                st.markdown("### Test Details:")
                                st.markdown(testing_results['test_details'])
            
            # Create standard visualizations
            figures = visualization.create_visualization(
                st.session_state.current_results, 
                metadata,
                column_types
            )
            
            # Create advanced visualizations based on data types
            advanced_figures = []
            
            # Create a heatmap for correlation analysis if there are multiple numerical columns
            if len(column_types['numerical']) >= 2:
                try:
                    heatmap_fig = visualization.create_advanced_visualization(
                        st.session_state.current_results,
                        'heatmap',
                        {},  # Will default to correlation heatmap
                        {
                            'title': 'Correlation Heatmap',
                            'color_theme': 'default',
                            'show_annotations': True
                        }
                    )
                    if heatmap_fig:
                        advanced_figures.append(("Correlation Heatmap", heatmap_fig))
                except Exception as e:
                    st.warning(f"Could not create correlation heatmap: {str(e)}")
            
            # Create a bubble chart if there are at least 3 numerical columns
            if len(column_types['numerical']) >= 3:
                try:
                    bubble_fig = visualization.create_advanced_visualization(
                        st.session_state.current_results,
                        'bubble',
                        {
                            'x': column_types['numerical'][0],
                            'y': column_types['numerical'][1],
                            'size': column_types['numerical'][2],
                            'color': column_types['categorical'][0] if column_types['categorical'] else None
                        },
                        {
                            'title': f'Bubble Chart: {column_types["numerical"][0]} vs {column_types["numerical"][1]}',
                            'color_theme': 'vibrant'
                        }
                    )
                    if bubble_fig:
                        advanced_figures.append(("Bubble Chart", bubble_fig))
                except Exception as e:
                    st.warning(f"Could not create bubble chart: {str(e)}")
                    
            # Create distribution plot for numerical columns
            if column_types['numerical']:
                try:
                    dist_fig = visualization.create_advanced_visualization(
                        st.session_state.current_results,
                        'distribution',
                        {
                            'value': column_types['numerical'][0],
                            'group': column_types['categorical'][0] if column_types['categorical'] else None
                        },
                        {
                            'title': f'Distribution of {column_types["numerical"][0]}',
                            'color_theme': 'light',
                            'show_rug': True
                        }
                    )
                    if dist_fig:
                        advanced_figures.append(("Distribution Plot", dist_fig))
                except Exception as e:
                    pass  # Skip if distribution couldn't be created
            
            # Create treemap if categorical and numerical columns exist
            if column_types['categorical'] and column_types['numerical'] and len(column_types['categorical']) >= 2:
                try:
                    treemap_fig = visualization.create_advanced_visualization(
                        st.session_state.current_results,
                        'treemap',
                        {
                            'path': column_types['categorical'][:2],
                            'values': column_types['numerical'][0],
                            'color': column_types['numerical'][0]
                        },
                        {
                            'title': f'Treemap of {column_types["numerical"][0]} by {", ".join(column_types["categorical"][:2])}',
                            'color_theme': 'dark'
                        }
                    )
                    if treemap_fig:
                        advanced_figures.append(("Treemap", treemap_fig))
                except Exception as e:
                    pass  # Skip if treemap couldn't be created
            
            # Add visualization tabs with both standard and advanced visualizations
            all_figures = [(f"Basic {i+1}", fig) for i, fig in enumerate(figures)]
            all_figures.extend(advanced_figures)
            
            if all_figures:
                # Create tabs with meaningful names
                tabs = st.tabs([name for name, _ in all_figures])
                for i, tab in enumerate(tabs):
                    with tab:
                        st.plotly_chart(all_figures[i][1], use_container_width=True)
                
                # Add custom visualization builder
                with st.expander("Create Custom Visualization"):
                    st.write("### Custom Visualization Builder")
                    
                    # Initialize visualization preferences if not already in session state
                    if 'viz_preferences' not in st.session_state:
                        st.session_state.viz_preferences = {
                            'color_theme': 'default',
                            'height': 600,
                            'show_legend': True
                        }
                    
                    # First row: Select visualization type
                    viz_types = [
                        "bar", "line", "pie", "scatter", "bubble", "heatmap", 
                        "treemap", "box", "histogram", "distribution", "radar",
                        "parallel_coordinates", "sunburst", "waterfall", "gantt"
                    ]
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        viz_type = st.selectbox(
                            "Select Visualization Type", 
                            viz_types, 
                            index=0, 
                            key="custom_viz_type"
                        )
                    
                    # Second row: Column selection based on visualization type
                    if viz_type in ["bar", "line", "scatter"]:
                        col1, col2 = st.columns(2)
                        with col1:
                            x_options = st.session_state.current_results.columns.tolist()
                            x_col = st.selectbox("X-Axis", x_options, key="custom_x_col")
                        with col2:
                            y_options = [col for col in st.session_state.current_results.columns if pd.api.types.is_numeric_dtype(st.session_state.current_results[col])]
                            if y_options:
                                y_col = st.selectbox("Y-Axis", y_options, key="custom_y_col")
                            else:
                                st.warning("No numerical columns available for Y-axis")
                                y_col = None
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            color_options = ["None"] + st.session_state.current_results.columns.tolist()
                            color_col = st.selectbox("Color By", color_options, key="custom_color_col")
                            color_col = None if color_col == "None" else color_col
                        
                        columns_dict = {"x": x_col, "y": y_col, "color": color_col}
                    
                    elif viz_type == "bubble":
                        num_columns = [col for col in st.session_state.current_results.columns if pd.api.types.is_numeric_dtype(st.session_state.current_results[col])]
                        
                        if len(num_columns) < 3:
                            st.warning("Bubble chart requires at least 3 numerical columns")
                            columns_dict = {}
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox("X-Axis", num_columns, key="custom_x_col")
                            with col2:
                                y_col = st.selectbox("Y-Axis", num_columns, key="custom_y_col")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                size_col = st.selectbox("Size", num_columns, key="custom_size_col")
                            with col2:
                                color_options = ["None"] + st.session_state.current_results.columns.tolist()
                                color_col = st.selectbox("Color By", color_options, key="custom_color_col")
                                color_col = None if color_col == "None" else color_col
                            
                            columns_dict = {"x": x_col, "y": y_col, "size": size_col, "color": color_col}
                    
                    elif viz_type == "pie":
                        col1, col2 = st.columns(2)
                        with col1:
                            cat_columns = [col for col in st.session_state.current_results.columns if col in column_types['categorical']]
                            if cat_columns:
                                names_col = st.selectbox("Names (Categories)", cat_columns, key="custom_names_col")
                            else:
                                st.warning("No categorical columns found for pie chart names")
                                names_col = None
                        
                        with col2:
                            num_columns = [col for col in st.session_state.current_results.columns if pd.api.types.is_numeric_dtype(st.session_state.current_results[col])]
                            if num_columns:
                                values_col = st.selectbox("Values", num_columns, key="custom_values_col")
                            else:
                                st.warning("No numerical columns found for pie chart values")
                                values_col = None
                        
                        columns_dict = {"names": names_col, "values": values_col}
                    
                    elif viz_type == "heatmap":
                        # For heatmap, we'll default to correlation matrix
                        columns_dict = {}
                    
                    elif viz_type in ["treemap", "sunburst"]:
                        cat_columns = [col for col in st.session_state.current_results.columns if col in column_types['categorical']]
                        num_columns = [col for col in st.session_state.current_results.columns if pd.api.types.is_numeric_dtype(st.session_state.current_results[col])]
                        
                        if not cat_columns:
                            st.warning(f"No categorical columns found for {viz_type}")
                            columns_dict = {}
                        else:
                            # Allow selecting multiple categorical columns for the path
                            path_cols = st.multiselect("Path (Hierarchy)", cat_columns, default=[cat_columns[0]] if cat_columns else [], key="custom_path_cols")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if num_columns:
                                    values_col = st.selectbox("Values", num_columns, key="custom_values_col")
                                else:
                                    st.warning("No numerical columns found for values")
                                    values_col = None
                            
                            with col2:
                                color_options = ["None"] + num_columns
                                color_col = st.selectbox("Color By", color_options, key="custom_color_col")
                                color_col = None if color_col == "None" else color_col
                            
                            columns_dict = {"path": path_cols, "values": values_col, "color": color_col}
                    
                    elif viz_type == "distribution":
                        num_columns = [col for col in st.session_state.current_results.columns if pd.api.types.is_numeric_dtype(st.session_state.current_results[col])]
                        
                        if not num_columns:
                            st.warning("No numerical columns found for distribution")
                            columns_dict = {}
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                value_col = st.selectbox("Value", num_columns, key="custom_value_col")
                            
                            with col2:
                                group_options = ["None"] + [col for col in st.session_state.current_results.columns if col in column_types['categorical']]
                                group_col = st.selectbox("Group By", group_options, key="custom_group_col")
                                group_col = None if group_col == "None" else group_col
                            
                            columns_dict = {"value": value_col, "group": group_col}
                    
                    elif viz_type == "box":
                        cat_columns = [col for col in st.session_state.current_results.columns if col in column_types['categorical']]
                        num_columns = [col for col in st.session_state.current_results.columns if pd.api.types.is_numeric_dtype(st.session_state.current_results[col])]
                        
                        if not cat_columns or not num_columns:
                            st.warning("Box plot requires both categorical and numerical columns")
                            columns_dict = {}
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox("X-Axis (Category)", cat_columns, key="custom_x_col")
                            with col2:
                                y_col = st.selectbox("Y-Axis (Value)", num_columns, key="custom_y_col")
                            
                            columns_dict = {"x": x_col, "y": y_col}
                    
                    elif viz_type == "histogram":
                        num_columns = [col for col in st.session_state.current_results.columns if pd.api.types.is_numeric_dtype(st.session_state.current_results[col])]
                        
                        if not num_columns:
                            st.warning("No numerical columns found for histogram")
                            columns_dict = {}
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox("Value", num_columns, key="custom_x_col")
                            
                            with col2:
                                color_options = ["None"] + [col for col in st.session_state.current_results.columns if col in column_types['categorical']]
                                color_col = st.selectbox("Group By", color_options, key="custom_color_col")
                                color_col = None if color_col == "None" else color_col
                            
                            columns_dict = {"x": x_col, "color": color_col}
                    
                    else:
                        # Default for other visualization types
                        st.info(f"Custom configuration for {viz_type} is not fully implemented. Using best guess settings.")
                        columns_dict = {}
                    
                    # Visual settings
                    st.write("#### Appearance Settings")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        theme_options = ["default", "light", "dark", "minimal", "business", "vibrant"]
                        selected_theme = st.selectbox(
                            "Color Theme", 
                            theme_options, 
                            index=theme_options.index(st.session_state.viz_preferences.get('color_theme', 'default')) if st.session_state.viz_preferences.get('color_theme', 'default') in theme_options else 0,
                            key="custom_theme"
                        )
                    with col2:
                        height = st.slider("Chart Height", min_value=300, max_value=1000, value=st.session_state.viz_preferences.get('height', 600), step=50, key="custom_height")
                    with col3:
                        show_legend = st.checkbox("Show Legend", value=st.session_state.viz_preferences.get('show_legend', True), key="custom_show_legend")
                    
                    # Title
                    title = st.text_input("Chart Title", key="custom_title", placeholder="Enter a title for your chart")
                    
                    # Settings dict
                    settings = {
                        'color_theme': selected_theme,
                        'height': height,
                        'show_legend': show_legend,
                        'title': title
                    }
                    
                    # Create custom visualization button
                    if st.button("Create Visualization", key="create_custom_viz"):
                        if columns_dict or viz_type in ["heatmap"]:
                            try:
                                custom_fig = visualization.create_advanced_visualization(
                                    st.session_state.current_results,
                                    viz_type,
                                    columns_dict,
                                    settings
                                )
                                
                                if custom_fig:
                                    st.plotly_chart(custom_fig, use_container_width=True)
                                    
                                    # Apply theme to custom figure
                                    custom_fig = visualization.apply_custom_theme(custom_fig, selected_theme)
                                    
                                    # Add to session state for persistence
                                    if 'custom_visualizations' not in st.session_state:
                                        st.session_state.custom_visualizations = []
                                    
                                    viz_name = title if title else f"Custom {viz_type.capitalize()}"
                                    st.session_state.custom_visualizations.append((viz_name, custom_fig))
                                    
                                    st.success(f"Created custom {viz_type} visualization!")
                                else:
                                    st.warning(f"Could not create {viz_type} visualization with the selected settings")
                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
                        else:
                            st.warning("Please select appropriate columns for this visualization type")
                
                # Add visualization customization options
                with st.expander("Global Visualization Settings"):
                    st.write("### Global Visualization Preferences")
                    col1, col2 = st.columns(2)
                    with col1:
                        theme_options = ["default", "light", "dark", "minimal", "business", "vibrant"]
                        selected_theme = st.selectbox("Color Theme", theme_options, index=0)
                    with col2:
                        height = st.slider("Chart Height", min_value=300, max_value=1000, value=600, step=50)
                    
                    st.info("Note: Apply these settings by re-running the query")
                    
                    # Store customization preferences in session state
                    st.session_state.viz_preferences = {
                        'color_theme': selected_theme,
                        'height': height,
                        'show_legend': True
                    }
                    
                # Show user's custom visualizations if available
                if 'custom_visualizations' in st.session_state and st.session_state.custom_visualizations:
                    st.subheader("Your Custom Visualizations")
                    
                    # Add the ability to delete visualizations
                    delete_viz = st.selectbox(
                        "Select a visualization to delete:",
                        ["None"] + [viz[0] for viz in st.session_state.custom_visualizations],
                        index=0
                    )
                    
                    if delete_viz != "None" and st.button("Delete Selected Visualization"):
                        # Find and remove the visualization
                        st.session_state.custom_visualizations = [
                            viz for viz in st.session_state.custom_visualizations 
                            if viz[0] != delete_viz
                        ]
                        st.success(f"Deleted visualization: {delete_viz}")
                        st.rerun()
                    
                    # Display custom visualizations
                    for viz_name, viz_fig in st.session_state.custom_visualizations:
                        st.write(f"#### {viz_name}")
                        st.plotly_chart(viz_fig, use_container_width=True)
            else:
                st.info("No visualizations available for this query")
                
            # Query suggestions
            st.header("Looking for insights?")
            suggestions = [
                "Can you show me trends over time?",
                "What are the top performers?",
                "How does data compare across different categories?",
                "Are there any outliers in the data?",
                "Show me the distribution of values"
            ]
            
            suggestion_cols = st.columns(len(suggestions))
            for i, col in enumerate(suggestion_cols):
                with col:
                    if st.button(suggestions[i], key=f"suggestion_{i}"):
                        if st.session_state.data_source_mode == "tables":
                            # Tables mode - include table names
                            st.session_state.current_query = f"{suggestions[i]} for {', '.join(st.session_state.selected_tables)}"
                        else:
                            # Reports mode - include report names 
                            st.session_state.current_query = f"{suggestions[i]} for {', '.join(st.session_state.selected_reports)}"
                        st.rerun()
                        
else:
    st.info("Please connect to your database using the sidebar to start exploring your data")

# Footer
st.markdown("---")
st.caption("Built by Sanadi Technologies Pvt Ltd")
