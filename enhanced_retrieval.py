"""
Enhanced Retrieval Module for DB Insights Explorer

This module provides advanced retrieval techniques for both website content and PDF documents.
It implements hybrid search, adaptive chunking, hierarchical retrieval, result re-ranking,
and self-query mechanisms to improve search quality.
"""

import os
import re
import time
import json
import logging
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from datetime import datetime

import pandas as pd
import numpy as np
import chromadb
from bs4 import BeautifulSoup
import tiktoken

# Import model clients for LLM-based improvement of retrieval
import model_clients

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRetrieval:
    """Base class for enhanced retrieval functionality"""
    
    def __init__(self, model_provider="openai"):
        """
        Initialize with specified model provider
        
        Args:
            model_provider: AI model provider to use for advanced retrieval techniques
        """
        self.model_provider = model_provider
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the LLM client when needed"""
        if self._client is None:
            self._client = model_clients.get_model_client(self.model_provider)
        return self._client

    def generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate multiple variations of a query to improve retrieval
        
        Args:
            query: Original user query
            num_variations: Number of query variations to generate
            
        Returns:
            List of query variations
        """
        if not self.client:
            logger.error("Unable to connect to AI model for query variation generation")
            return [query]  # Return original query if model client not available
        
        # Prompt to generate query variations
        prompt = f"""
        I need to find information to answer the question: "{query}"
        
        Generate {num_variations} different search queries that would help retrieve relevant information from a document database. 
        
        The queries should:
        1. Capture different aspects of the original question
        2. Use different phrasing and keywords
        3. Be specific enough to find relevant information
        4. Each be on a separate line
        
        Return ONLY the search queries, one per line, with no additional text or explanation.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates search query variations to improve document retrieval."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.generate_completion(messages, temperature=0.7)
            
            # Process response and extract queries
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Ensure we have at least one query (the original)
            if not variations:
                variations = [query]
            
            # Add the original query if it's not already in the list
            if query not in variations:
                variations.append(query)
                
            logger.info(f"Generated {len(variations)} query variations")
            return variations
        except Exception as e:
            logger.error(f"Error generating query variations: {str(e)}")
            return [query]  # Return original query if error

    def rerank_results(self, query: str, documents: List[str], scores: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """
        Re-rank search results using LLM scoring
        
        Args:
            query: Original query
            documents: List of document chunks to re-rank
            scores: Optional list of initial scores (from vector similarity)
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not self.client:
            # Return original order if model client not available
            if scores:
                return [(doc, score) for doc, score in zip(documents, scores)]
            return [(doc, 1.0 - i/len(documents)) for i, doc in enumerate(documents)]
        
        # Prompt for scoring each document
        prompt_template = """
        On a scale of 0 to 10, rate the relevance of the following text to the query: "{query}"
        
        Text: {document}
        
        Provide your rating as a single number (0-10) with NO additional explanation or text.
        """
        
        scored_documents = []
        for i, document in enumerate(documents):
            try:
                # Skip if document is too short to be meaningful
                if len(document.strip()) < 20:
                    continue
                    
                # Keep document short enough for context window
                doc_excerpt = document[:3000] + "..." if len(document) > 3000 else document
                
                prompt = prompt_template.format(query=query, document=doc_excerpt)
                messages = [
                    {"role": "system", "content": "You are an AI assistant that scores document relevance to a query on a scale of 0-10. Respond with ONLY a number."},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.client.generate_completion(messages, temperature=0.1)
                
                # Extract score - look for a number in the response
                score_match = re.search(r'\\b(\\d+(?:\\.\\d+)?)\\b', response)
                if score_match:
                    score = float(score_match.group(1))
                    # Normalize to 0-1
                    normalized_score = min(max(score / 10.0, 0.0), 1.0)
                    scored_documents.append((document, normalized_score))
                else:
                    # If no score found, use original score if available or add with neutral score
                    if scores and i < len(scores):
                        scored_documents.append((document, scores[i]))
                    else:
                        scored_documents.append((document, 0.5))
                    
            except Exception as e:
                logger.error(f"Error scoring document: {str(e)}")
                # Add with lower score on error or use original score if available
                if scores and i < len(scores):
                    scored_documents.append((document, scores[i]))
                else:
                    scored_documents.append((document, 0.3))
        
        # Sort by score in descending order
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents


class AdaptiveChunker:
    """Handles adaptive chunking for different content types"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks of specified size (default method)
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Clean the text
        text = text.replace("\r", " ")
        text_length = len(text)
        
        chunks = []
        start = 0
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # If we're not at the beginning, adjust start to create overlap
            if start > 0:
                start = start - overlap
                
            # Find a good break point (newline or space)
            if end < text_length:
                # Try to find a newline to break at
                newline_pos = text.rfind("\n", start, end)
                space_pos = text.rfind(" ", start, end)
                
                if newline_pos > start + chunk_size // 2:
                    end = newline_pos + 1  # Include the newline
                elif space_pos > start + chunk_size // 2:
                    end = space_pos + 1  # Include the space
            
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
                
            start = end
        
        return chunks
    
    @staticmethod
    def adaptive_chunking(text: str, content_type: str = "general", chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Adaptively chunk text based on content type
        
        Args:
            text: Text to split into chunks
            content_type: Type of content (general, article, technical, qa, pdf)
            chunk_size: Target chunk size
            overlap: Target overlap size
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Clean the text
        text = text.replace("\r", " ")
        
        # Define chunking parameters based on content type
        if content_type == "article":
            # For articles, prefer paragraph breaks
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            # Combine short paragraphs and split overly long ones
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) < chunk_size:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                else:
                    # If current chunk is not empty, add it to chunks
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Handle overly long paragraphs
                    if len(paragraph) > chunk_size + (chunk_size // 2):
                        # Split by sentences for long paragraphs
                        sentences = re.split(r'(?<=[.!?])\\s+', paragraph)
                        
                        current_chunk = ""
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) < chunk_size:
                                current_chunk += " " + sentence if current_chunk else sentence
                            else:
                                chunks.append(current_chunk)
                                current_chunk = sentence
                    else:
                        current_chunk = paragraph
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
                
        elif content_type == "technical":
            # For technical content, try to keep code blocks intact
            
            # Split by code block markers or headers
            patterns = [
                r'```.*?```',  # Code blocks
                r'#+\\s+.*?\\n',  # Headers
                r'\\n\\n'         # Double newlines
            ]
            
            # Create combined pattern
            combined_pattern = '|'.join(f'({p})' for p in patterns)
            
            # Split text but keep the delimiters
            tokens = re.split(f'({combined_pattern})', text)
            
            chunks = []
            current_chunk = ""
            
            for token in tokens:
                if not token.strip():
                    continue
                    
                # If adding this token would exceed the chunk size
                if len(current_chunk) + len(token) > chunk_size + (chunk_size // 5):
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Handle very long tokens
                    if len(token) > chunk_size + (chunk_size // 2):
                        # Further split long non-code tokens
                        if not token.startswith('```'):
                            for subchunk in [token[i:i+chunk_size] for i in range(0, len(token), chunk_size-overlap)]:
                                chunks.append(subchunk)
                        else:
                            # Keep long code blocks intact
                            chunks.append(token)
                            
                        current_chunk = ""
                    else:
                        current_chunk = token
                else:
                    current_chunk += token
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
                
        elif content_type == "qa":
            # For Q&A content, try to keep questions and answers together
            
            # Look for patterns like "Q:" or "Question:" followed by "A:" or "Answer:"
            qa_blocks = re.split(r'(?=\\b(?:Q:|Question:))', text)
            
            chunks = []
            current_chunk = ""
            
            for block in qa_blocks:
                if not block.strip():
                    continue
                    
                if len(current_chunk) + len(block) < chunk_size:
                    current_chunk += "\n\n" + block if current_chunk else block
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Handle overly long QA blocks
                    if len(block) > chunk_size + (chunk_size // 2):
                        # Try to split between question and answer if possible
                        answer_match = re.search(r'\\b(?:A:|Answer:)', block)
                        if answer_match and answer_match.start() > 200:
                            chunks.append(block[:answer_match.start()].strip())
                            current_chunk = block[answer_match.start():].strip()
                        else:
                            # Fall back to character-based chunks with overlap
                            subchunks = AdaptiveChunker.chunk_text(block, chunk_size, overlap)
                            chunks.extend(subchunks[:-1])
                            current_chunk = subchunks[-1] if subchunks else ""
                    else:
                        current_chunk = block
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
                
        elif content_type == "pdf":
            # For PDFs, try to keep pages together when possible and respect headers
            # Look for page markers, headers, and sections
            sections = re.split(r'(?:\\n\\s*#{1,3}\\s+|\\n\\s*Page\\s+\\d+|\\n\\s*Section\\s+\\d+)', text)
            
            if len(sections) <= 1:
                # If no clear section markers, fall back to paragraph splitting
                sections = [p for p in text.split("\n\n") if p.strip()]
                
            chunks = []
            current_chunk = ""
            
            for section in sections:
                if not section.strip():
                    continue
                    
                if len(current_chunk) + len(section) < chunk_size:
                    current_chunk += "\n\n" + section if current_chunk else section
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Handle very long sections
                    if len(section) > chunk_size + (chunk_size // 2):
                        # Try splitting by paragraphs first
                        paragraphs = section.split("\n\n")
                        
                        if len(paragraphs) > 1:
                            # If we can split into paragraphs, do so
                            current_chunk = ""
                            for para in paragraphs:
                                if len(current_chunk) + len(para) < chunk_size:
                                    current_chunk += "\n\n" + para if current_chunk else para
                                else:
                                    chunks.append(current_chunk)
                                    current_chunk = para
                        else:
                            # Fall back to sentence splitting for long paragraphs
                            subchunks = AdaptiveChunker.chunk_text(section, chunk_size, overlap)
                            chunks.extend(subchunks[:-1])
                            current_chunk = subchunks[-1] if subchunks else ""
                    else:
                        current_chunk = section
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
        else:
            # Default to the standard chunking method
            chunks = AdaptiveChunker.chunk_text(text, chunk_size, overlap)
        
        # Final check: break any chunks that are still too long
        result_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size + (chunk_size // 2):
                result_chunks.extend(AdaptiveChunker.chunk_text(chunk, chunk_size, overlap))
            else:
                result_chunks.append(chunk)
        
        return result_chunks


class WebContentRetrieval(EnhancedRetrieval):
    """Enhanced retrieval specifically for web content"""
    
    def __init__(self, model_provider="openai"):
        """Initialize with model provider"""
        super().__init__(model_provider)
        self.chunker = AdaptiveChunker()
    
    def hybrid_search(self, collection: chromadb.Collection, query: str, n_results: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Perform hybrid search using both keyword and semantic search
        
        Args:
            collection: ChromaDB collection
            query: Query string
            n_results: Number of results to return
            
        Returns:
            Tuple of (documents, metadatas, scores)
        """
        try:
            # 1. Semantic vector search
            semantic_results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            semantic_docs = semantic_results.get("documents", [[]])[0]
            semantic_meta = semantic_results.get("metadatas", [[]])[0]
            semantic_dist = semantic_results.get("distances", [[]])[0]
            
            # Convert distances to scores (1.0 is best, 0.0 is worst)
            semantic_scores = [1.0 - min(dist, 1.0) for dist in semantic_dist] if semantic_dist else []
            
            # 2. Keyword search by building our own where_document if filtering is supported
            key_terms = [term.lower() for term in query.split() if len(term) > 3]
            seen_docs = set()
            keyword_results = []
            
            # Try to get documents containing key terms if collection is big enough
            try:
                # First check collection size to avoid errors for small collections
                count_result = collection.count()
                
                # Only do keyword search if we have enough documents
                if count_result > n_results:
                    # Start by adding all semantic results to seen_docs so we don't duplicate
                    for doc in semantic_docs:
                        seen_docs.add(doc)
                    
                    # Try each key term to find more documents
                    for term in key_terms:
                        try:
                            # Replace with simpler query for compatibility
                            term_results = collection.query(
                                query_texts=[term], 
                                n_results=3
                            )
                            if term_results.get("documents"):
                                term_docs = term_results.get("documents", [[]])[0]
                                term_meta = term_results.get("metadatas", [[]])[0]
                                term_dist = term_results.get("distances", [[]])[0]
                                
                                term_scores = [1.0 - min(dist, 1.0) for dist in term_dist] if term_dist else []
                                
                                for i, doc in enumerate(term_docs):
                                    # Only add if not already in semantic results
                                    if doc not in seen_docs:
                                        seen_docs.add(doc)
                                        keyword_results.append((
                                            doc, 
                                            term_meta[i] if i < len(term_meta) else {},
                                            term_scores[i] if i < len(term_scores) else 0.5
                                        ))
                        except Exception as e:
                            logger.warning(f"Error in keyword search for term '{term}': {str(e)}")
                            continue
            except Exception as e:
                logger.warning(f"Error in keyword search setup: {str(e)}")
            
            # 3. Combine results
            combined_docs = list(semantic_docs)
            combined_meta = list(semantic_meta)
            combined_scores = list(semantic_scores)
            
            # Add keyword results (if any)
            for doc, meta, score in keyword_results:
                # Limit to requested size
                if len(combined_docs) >= n_results * 2:
                    break
                combined_docs.append(doc)
                combined_meta.append(meta)
                combined_scores.append(score * 0.9)  # Slightly lower weight for keyword matches
            
            # Return all results up to n_results or more if we have semantic and keyword results
            max_results = min(len(combined_docs), max(n_results, min(n_results*2, 10)))
            return combined_docs[:max_results], combined_meta[:max_results], combined_scores[:max_results]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fall back to standard query
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                scores = [1.0 - min(dist, 1.0) for dist in distances] if distances else []
                return documents, metadatas, scores
            except Exception as e2:
                logger.error(f"Error in fallback query: {str(e2)}")
                return [], [], []

    def hierarchical_retrieval(self, query: str, collection_name: str, n_results: int = 5, client = None) -> pd.DataFrame:
        """
        Implement hierarchical retrieval for better context preservation
        
        Args:
            query: Search query
            collection_name: Name of the collection to search
            n_results: Number of top chunks to retrieve
            client: Optional ChromaDB client (if not provided, a new one will be created)
            
        Returns:
            DataFrame with search results and their surrounding context
        """
        # Get client if not provided
        if client is None:
            from web_scraper import get_chroma_client, get_openai_embedding_function
            client = get_chroma_client()
            if not client:
                logger.error("Failed to initialize ChromaDB client")
                return pd.DataFrame()
                
            embedding_func = get_openai_embedding_function()
            if not embedding_func:
                logger.error("Failed to initialize OpenAI embedding function")
                return pd.DataFrame()
        else:
            # Assuming embedding function is already set in the client
            pass
            
        # Get collection
        try:
            from web_scraper import sanitize_collection_name
            sanitized_name = sanitize_collection_name(collection_name)
            collection = client.get_collection(name=sanitized_name)
        except Exception as e:
            logger.error(f"Error getting collection {collection_name}: {str(e)}")
            return pd.DataFrame()
        
        # First, use hybrid search to get the most relevant chunks
        documents, metadatas, scores = self.hybrid_search(collection, query, n_results)
        
        # For each result, also fetch surrounding chunks from the same source
        context_enriched_results = []
        
        for doc, meta, score in zip(documents, metadatas, scores):
            # Extract source and chunk information
            source = meta.get("source", "")
            chunk_id = meta.get("chunk_id", 0)
            
            # Create ID for current chunk
            current_id = f"{source}_{chunk_id}"
            
            # Find adjacent chunks (previous and next)
            surrounding_chunks = []
            
            # Try to get previous chunk
            if isinstance(chunk_id, (int, float)) and chunk_id > 0:
                prev_id = f"{source}_{chunk_id-1}"
                try:
                    prev_result = collection.get(ids=[prev_id])
                    if prev_result and prev_result.get("documents") and prev_result["documents"]:
                        prev_metadata = prev_result.get("metadatas", [{}])[0]
                        surrounding_chunks.append({
                            "content": prev_result["documents"][0],
                            "source": source,
                            "chunk_id": chunk_id-1,
                            "position": "previous",
                            "score": score * 0.85  # Lower score for context chunks
                        })
                        # Add other metadata
                        for k, v in prev_metadata.items():
                            if k not in ["source", "chunk_id"]:
                                surrounding_chunks[-1][k] = v
                except Exception as e:
                    logger.warning(f"Error getting previous chunk {prev_id}: {str(e)}")
            
            # Add the main chunk
            main_chunk = {
                "content": doc,
                "source": source,
                "chunk_id": chunk_id,
                "position": "main",
                "score": score
            }
            # Add other metadata
            for k, v in meta.items():
                if k not in ["source", "chunk_id"]:
                    main_chunk[k] = v
                    
            surrounding_chunks.append(main_chunk)
            
            # Try to get next chunk
            if isinstance(chunk_id, (int, float)):
                next_id = f"{source}_{chunk_id+1}"
                try:
                    next_result = collection.get(ids=[next_id])
                    if next_result and next_result.get("documents") and next_result["documents"]:
                        next_metadata = next_result.get("metadatas", [{}])[0]
                        surrounding_chunks.append({
                            "content": next_result["documents"][0],
                            "source": source,
                            "chunk_id": chunk_id+1,
                            "position": "next",
                            "score": score * 0.85  # Lower score for context chunks
                        })
                        # Add other metadata
                        for k, v in next_metadata.items():
                            if k not in ["source", "chunk_id"]:
                                surrounding_chunks[-1][k] = v
                except Exception as e:
                    logger.warning(f"Error getting next chunk {next_id}: {str(e)}")
            
            # Add to results
            context_enriched_results.extend(surrounding_chunks)
        
        # Create DataFrame and remove duplicates
        df = pd.DataFrame(context_enriched_results)
        if not df.empty:
            df = df.drop_duplicates(subset=["content"])
            df = df.sort_values(by="score", ascending=False)
            
        return df

    def self_query_mechanism(self, user_query: str, collection_name: str) -> pd.DataFrame:
        """
        Implement self-query mechanism that can translate natural language queries into
        structured filters + semantic search
        
        Args:
            user_query: Natural language query
            collection_name: Collection to search
            
        Returns:
            DataFrame with search results
        """
        if not self.client:
            # Fall back to standard search via hierarchical retrieval
            return self.hierarchical_retrieval(user_query, collection_name)
        
        # Prompt to extract structured query components
        prompt = f"""
        Analyze this search query: "{user_query}"
        
        Break it down into:
        1. Main search terms (what to search for semantically)
        2. Filters (specific constraints like date ranges, sources, or types of content)
        3. Sort preferences (how results should be ordered)
        
        Return your analysis as a JSON object with these keys:
        - semantic_query: The core semantic search terms
        - filters: Object with filter conditions (source, date_before, date_after, contains)
        - sort_by: What to sort by (relevance, date, etc.)
        
        Reply ONLY with the JSON, no other text.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a search query analyzer that extracts structured components from natural language queries."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.generate_completion(messages, temperature=0.1)
            
            # Parse the JSON response
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'({[\\s\\S]*})', response)
            if json_match:
                query_components = json.loads(json_match.group(1))
            else:
                # If JSON parsing fails, fall back to original query
                query_components = {"semantic_query": user_query}
                
            # Get the semantic component
            semantic_query = query_components.get("semantic_query", user_query)
            
            # Get any filters
            filters = query_components.get("filters", {})
            
            # Initialize ChromaDB client
            from web_scraper import get_chroma_client, get_openai_embedding_function
            client = get_chroma_client()
            if not client:
                return pd.DataFrame()
                
            embedding_func = get_openai_embedding_function()
            if not embedding_func:
                return pd.DataFrame()
                
            # Get collection
            from web_scraper import sanitize_collection_name
            sanitized_name = sanitize_collection_name(collection_name)
            
            try:
                collection = client.get_collection(name=sanitized_name, embedding_function=embedding_func)
            except Exception as e:
                logger.error(f"Error getting collection: {str(e)}")
                return pd.DataFrame()
                
            # Build query parameters
            query_params = {
                "query_texts": [semantic_query],
                "n_results": 10,
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Execute query (we'll filter post-query since ChromaDB has limited where filters)
            results = collection.query(**query_params)
            
            # Process results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            scores = [1.0 - min(dist, 1.0) for dist in distances] if distances else []
            
            # Post-query filtering
            filtered_results = []
            for i, (doc, meta, score) in enumerate(zip(documents, metadatas, scores)):
                # Check if this document matches the filters
                include_doc = True
                
                # Source filter
                if "source" in filters and filters["source"] and meta.get("source"):
                    if filters["source"].lower() not in meta.get("source", "").lower():
                        include_doc = False
                        
                # Date filters
                if "date_after" in filters and filters["date_after"] and "timestamp" in meta:
                    try:
                        doc_time = float(meta["timestamp"])
                        filter_time = time.mktime(datetime.strptime(filters["date_after"], "%Y-%m-%d").timetuple())
                        if doc_time < filter_time:
                            include_doc = False
                    except:
                        pass
                        
                if "date_before" in filters and filters["date_before"] and "timestamp" in meta:
                    try:
                        doc_time = float(meta["timestamp"])
                        filter_time = time.mktime(datetime.strptime(filters["date_before"], "%Y-%m-%d").timetuple())
                        if doc_time > filter_time:
                            include_doc = False
                    except:
                        pass
                
                # Content filter
                if "contains" in filters and filters["contains"]:
                    if filters["contains"].lower() not in doc.lower():
                        include_doc = False
                
                # Add document if it passes all filters
                if include_doc:
                    result_data = {
                        "content": doc,
                        "score": score
                    }
                    # Add all metadata
                    for k, v in meta.items():
                        result_data[k] = v
                        
                    filtered_results.append(result_data)
            
            # Create DataFrame
            df = pd.DataFrame(filtered_results)
            
            if df.empty:
                # If no results after filtering, fall back to regular search
                return self.hierarchical_retrieval(user_query, collection_name)
            
            # Apply sorting if specified
            sort_by = query_components.get("sort_by", "relevance")
            if sort_by == "date" and "timestamp" in df.columns:
                df = df.sort_values(by="timestamp", ascending=False)
            else:
                df = df.sort_values(by="score", ascending=False)
                
            return df
                
        except Exception as e:
            logger.error(f"Error in self-query mechanism: {str(e)}")
            # Fall back to hierarchical retrieval
            return self.hierarchical_retrieval(user_query, collection_name)

    def enhanced_web_search(self, query: str, collection_name: Union[str, List[str]], n_results: int = 5) -> pd.DataFrame:
        """
        Comprehensive enhanced search for web content with all improvements
        
        Args:
            query: User's query
            collection_name: Collection name or list of collections
            n_results: Number of results per collection
            
        Returns:
            DataFrame with search results
        """
        # Convert single collection to list for consistent processing
        collections = collection_name if isinstance(collection_name, list) else [collection_name]
        
        # Generate query variations for better recall
        query_variations = self.generate_query_variations(query, num_variations=3)
        
        all_results_dfs = []
        
        # Search each collection with each query variation
        for collection in collections:
            # Try self-query first for the original query
            try:
                self_query_df = self.self_query_mechanism(query, collection)
                if not self_query_df.empty:
                    self_query_df['collection'] = collection
                    self_query_df['query_variation'] = 'self_query'
                    all_results_dfs.append(self_query_df)
                    continue  # Skip other query variations if self-query is successful
            except Exception as e:
                logger.warning(f"Self-query mechanism failed for {collection}: {str(e)}")
            
            # Fall back to query variations with hierarchical retrieval
            for variation in query_variations:
                try:
                    df = self.hierarchical_retrieval(variation, collection, n_results)
                    if not df.empty:
                        df['collection'] = collection
                        df['query_variation'] = variation
                        all_results_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Hierarchical retrieval failed for {collection} with query '{variation}': {str(e)}")
        
        # Combine all results
        if not all_results_dfs:
            return pd.DataFrame()  # No results
            
        combined_df = pd.concat(all_results_dfs, ignore_index=True)
        
        # Remove duplicates, keeping the highest scoring version of each chunk
        if not combined_df.empty:
            combined_df = combined_df.sort_values('score', ascending=False)
            combined_df = combined_df.drop_duplicates(subset=['content'])
            
            # Rerank the top results for better precision
            if len(combined_df) > 0:
                try:
                    top_docs = combined_df['content'].tolist()[:min(len(combined_df), 10)]
                    top_scores = combined_df['score'].tolist()[:min(len(combined_df), 10)]
                    
                    # Rerank
                    reranked_docs = self.rerank_results(query, top_docs, top_scores)
                    
                    # Update scores in the dataframe for the reranked documents
                    score_map = {doc: score for doc, score in reranked_docs}
                    
                    # Create a new column for reranked scores
                    combined_df['reranked_score'] = combined_df['content'].map(
                        lambda x: score_map.get(x, 0.0) if x in score_map else combined_df.loc[combined_df['content'] == x, 'score'].values[0]
                    )
                    
                    # Sort by reranked score
                    combined_df = combined_df.sort_values('reranked_score', ascending=False)
                    
                except Exception as e:
                    logger.warning(f"Reranking failed: {str(e)}")
        
        return combined_df


class PDFContentRetrieval(EnhancedRetrieval):
    """Enhanced retrieval specifically for PDF content"""
    
    def __init__(self, model_provider="openai"):
        """Initialize with model provider"""
        super().__init__(model_provider)
        self.chunker = AdaptiveChunker()
    
    def enhanced_pdf_search(self, query: str, collection_ids: List[str], use_reranking: bool = True) -> List[Dict]:
        """
        Enhanced search for PDF content combining multiple techniques
        
        Args:
            query: The user's query
            collection_ids: List of PDF collection IDs to search
            use_reranking: Whether to use LLM-based reranking
            
        Returns:
            List of document chunks with metadata
        """
        # First check if PDF modules are available
        try:
            from pdf_processor import PDFProcessor
            pdf_processor = PDFProcessor(self.model_provider)
        except ImportError:
            logger.error("PDF processor module not available")
            return []
            
        # Get the standard retriever
        try:
            # Log the retrieval process for debugging
            logger.info(f"Retrieving documents from {len(collection_ids)} PDF collections using enhanced retrieval")
            
            # Get the base retriever using our improved implementation
            retriever = pdf_processor.get_retriever(collection_ids, use_multiquery=False)
            
            # Generate query variations
            query_variations = self.generate_query_variations(query, num_variations=2)
            logger.info(f"Generated {len(query_variations)} query variations for enhanced retrieval")
            
            # Use BaseRetriever implementation that now works correctly
            all_docs = []
            
            # First use the default retriever with original query
            try:
                # Typically retrieves 4 docs per collection
                logger.info(f"Retrieving documents with original query: {query}")
                # Always use get_relevant_documents method instead of directly accessing retrieval_function
                retrieved_docs = retriever.get_relevant_documents(query)
                logger.info(f"Retrieved {len(retrieved_docs)} documents with original query")
                all_docs.extend(retrieved_docs)
            except Exception as e:
                logger.warning(f"Error using default retriever: {str(e)}")
                logger.warning(f"Retriever type: {type(retriever)}")
                
            # Then search using each variation manually to ensure we get good coverage
            for variation in query_variations:
                if variation == query:  # Skip if same as original query
                    continue
                    
                try:
                    # Use direct retrieval for variations
                    logger.info(f"Retrieving documents with query variation: {variation}")
                    # Always use get_relevant_documents method instead of trying to access retrieval_function
                    var_docs = retriever.get_relevant_documents(variation)
                    logger.info(f"Retrieved {len(var_docs)} documents with variation")
                    all_docs.extend(var_docs)
                except Exception as e:
                    logger.warning(f"Error retrieving with variation '{variation}': {str(e)}")
                    logger.warning(f"Retriever type: {type(retriever)}")
            
            # Remove duplicates while preserving order
            unique_docs = []
            seen_page_content = set()
            
            for doc in all_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_page_content:
                    seen_page_content.add(content_hash)
                    unique_docs.append(doc)
            
            logger.info(f"After deduplication: {len(unique_docs)} unique documents")
            
            # Initialize results list
            results = []
            
            # First process docs to extract metadata
            for doc in unique_docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 0.5  # Default score, will be overridden by reranking
                }
                
                # Extract metadata
                for key, value in doc.metadata.items():
                    result[key] = value
                    
                results.append(result)
                
            # Rerank documents if requested
            if use_reranking and len(results) > 1:
                docs_to_rerank = [r["content"] for r in results]
                
                try:
                    reranked_docs = self.rerank_results(query, docs_to_rerank)
                    
                    # Update scores in results
                    for i, (doc, score) in enumerate(reranked_docs):
                        for j, result in enumerate(results):
                            if result["content"] == doc:
                                results[j]["score"] = score
                                break
                    
                    # Sort by score
                    results.sort(key=lambda x: x["score"], reverse=True)
                    
                except Exception as e:
                    logger.warning(f"Error in reranking: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced PDF search: {str(e)}")
            return []