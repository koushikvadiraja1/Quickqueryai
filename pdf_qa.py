"""
PDF Question Answering Module
This module handles the question answering functionality for PDF documents
using LLMs and Retrieval Augmented Generation (RAG) techniques.
"""

import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
import os
import time
import re
import json

import model_clients
from pdf_processor import PDFProcessor, highlight_text_in_context, extract_query_terms

class PDFQuestionAnswering:
    """Handles question answering for PDF documents"""
    
    def __init__(self, model_provider="openai", model_name=None):
        """
        Initialize PDF QA with the specified model provider
        
        Args:
            model_provider: The AI model provider (openai, anthropic, mistral, ollama)
            model_name: Specific model to use for Q&A
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.pdf_processor = PDFProcessor(model_provider, model_name)
        
        # Define a robust error handler class that extends OpenAIClient
        class ErrorHandlingClient(model_clients.OpenAIClient):
            """A special fallback client for error handling"""
            def __init__(self, error_message):
                # No need to call super().__init__ which could fail
                self.model_name = "error-model"
                self.error_message = error_message
                # Initialize client as None so no actual API calls are made
                self.client = None
                self.api_key = None
            
            def generate_completion(self, **kwargs):
                """Return error message instead of making API call"""
                return f"Error: {self.error_message}"
            
            def generate_streaming_completion(self, **kwargs):
                """Return error message as a stream"""
                yield f"Error: {self.error_message}"
        
        try:
            # Get model client for Q&A
            self.model_client = model_clients.get_model_client(model_provider, model_name)
        except Exception as e:
            error_msg = f"Error initializing model client: {str(e)}"
            st.error(error_msg)
            
            # Create fallback OpenAI client
            try:
                self.model_client = model_clients.get_model_client("openai", "gpt-4o")
                st.warning("Using fallback OpenAI model. Check your API settings.")
            except Exception as e2:
                fallback_error = f"Could not initialize model client. Please check your API keys in the sidebar settings. Error: {str(e2)}"
                st.error(fallback_error)
                
                # Use our robust error handler instead
                self.model_client = ErrorHandlingClient(fallback_error)
    
    def answer_question(self, query: str, collection_ids: List[str], 
                       use_multiquery: bool = True, use_enhanced_retrieval: bool = True) -> Dict[str, Any]:
        """
        Answer a question about the selected PDFs using RAG with enhanced retrieval
        
        Args:
            query: User's question
            collection_ids: List of PDF collection IDs to search
            use_multiquery: Whether to use multiquery retrieval for better results
            use_enhanced_retrieval: Whether to use advanced retrieval techniques
            
        Returns:
            Dictionary with answer and source information
        """
        if not collection_ids:
            return {
                "answer": "No PDF documents selected. Please upload and select at least one document.",
                "sources": [],
                "query_time": 0
            }
        
        start_time = time.time()
        
        try:
            # Try using enhanced retrieval if available
            if use_enhanced_retrieval:
                try:
                    print(f"Using enhanced PDF content retrieval for query: {query}")
                    from enhanced_retrieval import PDFContentRetrieval
                    
                    # Initialize enhanced retrieval with the same model provider
                    enhanced_retriever = PDFContentRetrieval(model_provider=self.model_provider)
                    
                    # Get enhanced results with additional logging
                    print(f"Searching across {len(collection_ids)} PDF collections...")
                    results = enhanced_retriever.enhanced_pdf_search(query, collection_ids, use_reranking=True)
                    print(f"Enhanced retrieval found {len(results)} results")
                    
                    if results:
                        # Convert to Document objects for consistent processing
                        from langchain_core.documents import Document
                        docs = []
                        
                        for result in results:
                            # Create a Document object with the content and metadata
                            doc = Document(
                                page_content=result.get("content", ""),
                                metadata=result.get("metadata", {})
                            )
                            # Add any other metadata fields if present
                            for key, value in result.items():
                                if key not in ["content", "metadata"]:
                                    doc.metadata[key] = value
                                    
                            docs.append(doc)
                        
                        print(f"Successfully processed {len(docs)} documents from enhanced retrieval")
                    else:
                        # Fall back to regular retrieval
                        print("Enhanced retrieval returned no results, falling back to standard retrieval")
                        retriever = self.pdf_processor.get_retriever(collection_ids, use_multiquery)
                        docs = retriever.get_relevant_documents(query)
                        print(f"Standard retrieval found {len(docs)} documents")
                        
                except (ImportError, Exception) as e:
                    # Fall back to regular retrieval if enhanced retrieval fails
                    print(f"Enhanced retrieval failed, falling back to standard retrieval: {str(e)}")
                    retriever = self.pdf_processor.get_retriever(collection_ids, use_multiquery)
                    docs = retriever.get_relevant_documents(query)
                    print(f"Standard retrieval found {len(docs)} documents after enhanced retrieval failed")
            else:
                # Standard retrieval (enhanced retrieval disabled)
                print("Using standard PDF retrieval (enhanced retrieval disabled)")
                retriever = self.pdf_processor.get_retriever(collection_ids, use_multiquery)
                docs = retriever.get_relevant_documents(query)
                print(f"Standard retrieval found {len(docs)} documents")
            
            if not docs:
                return {
                    "answer": "No relevant information found in the selected documents.",
                    "sources": [],
                    "query_time": time.time() - start_time
                }
            
            # Format documents for context
            context_docs = []
            for i, doc in enumerate(docs):
                # Get page number and filename from metadata
                page_number = doc.metadata.get("page_number", "Unknown")
                filename = doc.metadata.get("filename", "Unknown")
                collection_id = doc.metadata.get("collection_id", "")
                
                # Format as a numbered source
                source_info = f"[Source {i+1}] From \"{filename}\", Page {page_number}"
                context_docs.append(f"{source_info}:\n{doc.page_content}\n")
            
            # Combine into a single context string
            context = "\n".join(context_docs)
            
            # Prepare the prompt for the model
            system_prompt = """You are a helpful AI assistant that answers questions based on PDF documents. 
Your task is to provide accurate, relevant answers based solely on the provided document extracts.

Guidelines:
1. Only use information from the provided document extracts to answer the question
2. If the information is not in the documents, say "I couldn't find information about this in the provided documents."
3. Cite your sources by referring to the source numbers in your answer (e.g., "According to [Source 1]...")
4. Be concise but thorough
5. If extracts from multiple pages/documents contain relevant information, synthesize it into a complete answer
6. Use quotes for direct citations"""
            
            user_prompt = f"""Question: {query}

Document extracts:
{context}

Please provide a comprehensive answer to the question, citing the relevant sources."""
            
            # Generate the answer using the model
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            answer = self.model_client.generate_completion(messages=messages)
            
            # Extract source information for highlighting
            sources = []
            for i, doc in enumerate(docs):
                collection_id = doc.metadata.get("collection_id", "")
                page_number = doc.metadata.get("page_number", "Unknown")
                filename = doc.metadata.get("filename", "Unknown")
                
                # Extract full page text for highlighting
                page_text = self.pdf_processor.extract_page_text(collection_id, page_number)
                
                # Get page image with robust error handling
                try:
                    page_image = self.pdf_processor.get_page_image(collection_id, page_number)
                    print(f"DEBUG: Got page image for {collection_id}, page {page_number}, length: {len(page_image) if page_image else 0}")
                except Exception as e:
                    print(f"ERROR: Failed to get page image for {collection_id}, page {page_number}: {str(e)}")
                    page_image = ""
                
                # Extract key terms from the query for highlighting
                query_terms = extract_query_terms(query)
                
                # Get highlighted snippets
                snippets = highlight_text_in_context(page_text, query_terms)
                for snippet in snippets:
                    snippet["page"] = page_number
                
                # Create source entry with all required fields
                source_entry = {
                    "source_id": i + 1,
                    "collection_id": collection_id,
                    "filename": filename,
                    "page": page_number,  # Add page field for consistency
                    "page_number": page_number,
                    "page_image": page_image,
                    "text": doc.page_content,  # Add text content for display
                    "snippets": snippets
                }
                
                # Debug page image data
                print(f"DEBUG: Source {i+1} collection {collection_id}, page {page_number}, image length: {len(page_image) if page_image else 0}")
                
                sources.append(source_entry)
            
            return {
                "answer": answer,
                "sources": sources,
                "query_time": time.time() - start_time
            }
        
        except Exception as e:
            st.error(f"Error answering question: {e}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "query_time": time.time() - start_time
            }
    
    def stream_answer(self, query: str, collection_ids: List[str], 
                     use_multiquery: bool = True, use_enhanced_retrieval: bool = True,
                     streamlit_placeholder=None):
        """
        Stream an answer to a question about the selected PDFs
        
        Args:
            query: User's question
            collection_ids: List of PDF collection IDs to search
            use_multiquery: Whether to use multiquery retrieval for better results
            use_enhanced_retrieval: Whether to use advanced retrieval techniques
            streamlit_placeholder: Streamlit placeholder for streaming output
        
        Returns:
            Dictionary with sources information
        """
        if not streamlit_placeholder:
            answer_result = self.answer_question(query, collection_ids, use_multiquery, use_enhanced_retrieval)
            return answer_result
        
        # Display a loading message in the placeholder
        streamlit_placeholder.markdown("Searching documents...")
        
        start_time = time.time()
        sources = []
        
        try:
            # Try to use enhanced retrieval if available
            if use_enhanced_retrieval:
                streamlit_placeholder.markdown("Using enhanced retrieval system...")
                try:
                    from enhanced_retrieval import PDFContentRetrieval
                    
                    # Initialize enhanced retrieval with the same model provider
                    enhanced_retriever = PDFContentRetrieval(model_provider=self.model_provider)
                    
                    # Get enhanced results with additional logging
                    print(f"Streaming: Searching across {len(collection_ids)} PDF collections...")
                    streamlit_placeholder.markdown("Retrieving relevant document sections...")
                    results = enhanced_retriever.enhanced_pdf_search(query, collection_ids, use_reranking=True)
                    print(f"Streaming: Enhanced retrieval found {len(results)} results")
                    
                    if results:
                        # Convert to Document objects for consistent processing
                        from langchain_core.documents import Document
                        docs = []
                        
                        for result in results:
                            # Create a Document object with the content and metadata
                            doc = Document(
                                page_content=result.get("content", ""),
                                metadata=result.get("metadata", {})
                            )
                            # Add any other metadata fields if present
                            for key, value in result.items():
                                if key not in ["content", "metadata"]:
                                    doc.metadata[key] = value
                                    
                            docs.append(doc)
                        
                        print(f"Streaming: Successfully processed {len(docs)} documents from enhanced retrieval")
                    else:
                        # Fall back to regular retrieval
                        streamlit_placeholder.markdown("Enhanced retrieval found no results, falling back to standard retrieval...")
                        print("Streaming: Enhanced retrieval returned no results, falling back to standard retrieval")
                        retriever = self.pdf_processor.get_retriever(collection_ids, use_multiquery)
                        docs = retriever.get_relevant_documents(query)
                        print(f"Streaming: Standard retrieval found {len(docs)} documents")
                        
                except (ImportError, Exception) as e:
                    # Fall back to regular retrieval if enhanced retrieval fails
                    streamlit_placeholder.markdown("Enhanced retrieval unavailable, using standard retrieval...")
                    print(f"Streaming: Enhanced retrieval failed, falling back to standard retrieval: {str(e)}")
                    print(f"Streaming: Error details: {type(e).__name__}")
                    retriever = self.pdf_processor.get_retriever(collection_ids, use_multiquery)
                    docs = retriever.get_relevant_documents(query)
                    print(f"Streaming: Standard retrieval found {len(docs)} documents after enhanced retrieval failed")
            else:
                # Get retriever for selected collections
                retriever = self.pdf_processor.get_retriever(collection_ids, use_multiquery)
                
                # Retrieve relevant documents
                streamlit_placeholder.markdown("Retrieving relevant document sections...")
                docs = retriever.get_relevant_documents(query)
            
            if not docs:
                streamlit_placeholder.markdown("No relevant information found in the selected documents.")
                return {
                    "sources": [],
                    "query_time": time.time() - start_time
                }
            
            # Format documents for context
            context_docs = []
            for i, doc in enumerate(docs):
                # Get page number and filename from metadata
                page_number = doc.metadata.get("page_number", "Unknown")
                filename = doc.metadata.get("filename", "Unknown")
                collection_id = doc.metadata.get("collection_id", "")
                
                # Format as a numbered source
                source_info = f"[Source {i+1}] From \"{filename}\", Page {page_number}"
                context_docs.append(f"{source_info}:\n{doc.page_content}\n")
                
                # Extract full page text for highlighting
                page_text = self.pdf_processor.extract_page_text(collection_id, page_number)
                
                # Get page image with robust error handling
                try:
                    page_image = self.pdf_processor.get_page_image(collection_id, page_number)
                    print(f"DEBUG: Got page image for {collection_id}, page {page_number}, length: {len(page_image) if page_image else 0}")
                except Exception as e:
                    print(f"ERROR: Failed to get page image for {collection_id}, page {page_number}: {str(e)}")
                    page_image = ""
                
                # Extract key terms from the query for highlighting
                query_terms = extract_query_terms(query)
                
                # Get highlighted snippets
                snippets = highlight_text_in_context(page_text, query_terms)
                for snippet in snippets:
                    snippet["page"] = page_number
                
                # Create source entry with all required fields
                source_entry = {
                    "source_id": i + 1,
                    "collection_id": collection_id,
                    "filename": filename,
                    "page": page_number,  # Add page field for consistency
                    "page_number": page_number,
                    "page_image": page_image,
                    "text": doc.page_content,  # Add text content for display
                    "snippets": snippets
                }
                
                # Debug page image data
                print(f"DEBUG: Source {i+1} collection {collection_id}, page {page_number}, image length: {len(page_image) if page_image else 0}")
                
                sources.append(source_entry)
            
            # Combine into a single context string
            context = "\n".join(context_docs)
            
            # Prepare the prompt for the model
            system_prompt = """You are a helpful AI assistant that answers questions based on PDF documents. 
Your task is to provide accurate, relevant answers based solely on the provided document extracts.

Guidelines:
1. Only use information from the provided document extracts to answer the question
2. If the information is not in the documents, say "I couldn't find information about this in the provided documents."
3. Cite your sources by referring to the source numbers in your answer (e.g., "According to [Source 1]...")
4. Be concise but thorough
5. If extracts from multiple pages/documents contain relevant information, synthesize it into a complete answer
6. Use quotes for direct citations"""
            
            user_prompt = f"""Question: {query}

Document extracts:
{context}

Please provide a comprehensive answer to the question, citing the relevant sources."""
            
            # Generate the answer using the model with streaming
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Clear placeholder and prepare for streaming
            streamlit_placeholder.empty()
            stream_container = streamlit_placeholder.container()
            stream_container.markdown("Generating answer...")
            
            # Stream the response
            try:
                answer_text = ""
                
                # Get the streaming generator
                stream_gen = self.model_client.generate_streaming_completion(messages=messages)
                
                # Create a placeholder for accumulating text
                stream_container.empty()
                answer_placeholder = stream_container.empty()
                
                for chunk in stream_gen:
                    # Append the new chunk to the accumulated text
                    answer_text += chunk
                    
                    # Replace the entire content instead of appending
                    answer_placeholder.markdown(answer_text)
                
                # Store the answer text in the placeholder attribute for retrieval by app.py
                streamlit_placeholder.answer_text = answer_text
                
                # We need to clear the placeholder's visible content since we'll display the processed answer with links later
                streamlit_placeholder.empty()
                
                # Return sources as dictionary for consistent API
                return {
                    "sources": sources,
                    "query_time": time.time() - start_time
                }
            
            except Exception as e:
                st.error(f"Error streaming answer: {e}")
                
                try:
                    # Fallback to non-streaming approach
                    answer = self.model_client.generate_completion(messages=messages)
                    if stream_container:
                        stream_container.empty()
                        stream_container.markdown(answer)
                    
                    return {
                        "answer": answer,
                        "sources": sources,
                        "query_time": time.time() - start_time
                    }
                except Exception as e2:
                    error_msg = f"Failed to generate answer: {str(e2)}"
                    st.error(error_msg)
                    if stream_container:
                        stream_container.error(error_msg)
                    
                    return {
                        "answer": error_msg,
                        "sources": sources,
                        "query_time": time.time() - start_time
                    }
        
        except Exception as e:
            st.error(f"Error answering question: {e}")
            streamlit_placeholder.markdown(f"An error occurred: {str(e)}")
            
            return {
                "sources": [],
                "query_time": time.time() - start_time
            }