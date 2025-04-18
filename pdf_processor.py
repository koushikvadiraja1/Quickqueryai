"""
PDF Processor Module
This module handles PDF document uploading, parsing, vectorizing, and querying functionality.
It supports complex PDF documents and implements advanced retrieval techniques.
"""

import os
import json
import tempfile
import uuid
import base64
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
import numpy as np
import pdfplumber
import pdf2image
from PIL import Image
from io import BytesIO
import pypdf  # For PDF manipulation

# LangChain imports
from langchain_qdrant import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
# Import MultiQueryRetriever from the correct location
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import model_clients

# PDF chunking and processing
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector database client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# Ensure paths exist
UPLOAD_DIR = "uploaded_pdfs"
VECTOR_DB_DIR = "qdrant_db"
PDF_METADATA_FILE = "pdf_metadata.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Initialize Qdrant client (local by default)
qdrant_client = None
try:
    # First try to connect with a bit more time for startup
    for attempt in range(3):
        try:
            qdrant_client = QdrantClient(path=VECTOR_DB_DIR, timeout=10.0)
            # Test the connection by getting collections
            qdrant_client.get_collections()
            break  # If we get here, connection works
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise
            import time
            time.sleep(1)  # Wait before retry
            
    if qdrant_client is None:
        raise ValueError("Failed to initialize Qdrant client after retries")
        
except Exception as e:
    st.error(f"Failed to initialize Qdrant client: {e}")
    qdrant_client = None
    
# Check the connection
if qdrant_client:
    try:
        collections = qdrant_client.get_collections()
        st.info(f"Connected to Qdrant with {len(collections.collections)} collections.")
    except Exception as e:
        st.error(f"Qdrant connection test failed: {e}")
        qdrant_client = None


class PDFProcessor:
    """PDF document processing, vectorization, and retrieval"""
    
    def __init__(self, model_provider="openai", model_name=None):
        """
        Initialize the PDF processor with the specified model provider
        
        Args:
            model_provider: The AI model provider (openai, anthropic, mistral, ollama)
            model_name: Specific model to use for embeddings and retrieval
        """
        self.model_provider = model_provider
        self.model_name = model_name
        
        # Use OpenAI embeddings by default (could be extended to support others)
        self.embeddings = OpenAIEmbeddings()
        
        # Create a vector store
        self.vector_store = None
        
        # Cache for page images and content
        self.pdf_page_images = {}
        
        # Load PDF metadata from disk if available
        self.pdf_metadata = load_pdf_metadata()
    
    def process_pdf(self, pdf_file, filename: str) -> str:
        """
        Process a PDF file, extract content, and store in vector database
        
        Args:
            pdf_file: The PDF file object from Streamlit uploader
            filename: Original filename
            
        Returns:
            Collection ID for this PDF
        """
        # Generate a unique ID for this document
        collection_id = f"pdf_{uuid.uuid4().hex}"
        
        # Create uploaded_pdfs directory if it doesn't exist
        os.makedirs("uploaded_pdfs", exist_ok=True)
        
        # Save PDF to a permanent location in the uploaded_pdfs directory
        pdf_save_path = os.path.join("uploaded_pdfs", f"{collection_id}.pdf")
        abs_pdf_path = os.path.abspath(pdf_save_path)
        print(f"Saving PDF to: {abs_pdf_path}")
        
        try:
            with open(pdf_save_path, "wb") as pdf_file_disk:
                pdf_file_disk.write(pdf_file.getvalue())
                
            # Verify the file exists
            if os.path.exists(pdf_save_path):
                print(f"PDF saved successfully at {pdf_save_path}, size: {os.path.getsize(pdf_save_path)} bytes")
            else:
                print(f"WARNING: Failed to save PDF at {pdf_save_path}")
        except Exception as e:
            print(f"ERROR saving PDF: {str(e)}")
        
        try:
            # Extract text with PyPDF
            loader = PyPDFLoader(pdf_save_path)
            documents = loader.load()
            
            # Store original page mapping and enhance metadata
            for i, doc in enumerate(documents):
                # Ensure page numbers start at 1 for user-facing content
                page_number = i + 1
                doc.metadata["page_number"] = page_number
                doc.metadata["filename"] = filename
                doc.metadata["collection_id"] = collection_id
                doc.metadata["path"] = pdf_save_path
            
            # Split documents into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""],
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Store metadata about the PDF
            self.pdf_metadata[collection_id] = {
                "filename": filename,
                "total_pages": len(documents),
                "chunks": len(split_docs),
                "path": pdf_save_path
            }
            
            # Extract images from PDF pages for display
            self.pdf_page_images[collection_id] = {}
            
            # Use pdf2image to convert pages to images
            try:
                images = pdf2image.convert_from_path(pdf_save_path)
                for i, img in enumerate(images):
                    # Convert PIL image to base64 for display
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    page_number = i + 1  # 1-indexed for user-facing content
                    self.pdf_page_images[collection_id][page_number] = img_str
            except Exception as e:
                st.warning(f"Could not extract page images: {e}")
            
            # Make sure the collection exists first
            try:
                # Check if Qdrant client is available
                if qdrant_client is None:
                    raise ValueError("Qdrant client is not initialized")
                    
                # Check if collection exists
                collection_names = [c.name for c in qdrant_client.get_collections().collections]
                if collection_id not in collection_names:
                    # Create the collection with the right dimensions
                    embedding_dimension = 1536  # Default for OpenAI embeddings
                    qdrant_client.create_collection(
                        collection_name=collection_id,
                        vectors_config=qdrant_models.VectorParams(
                            size=embedding_dimension,
                            distance=qdrant_models.Distance.COSINE
                        )
                    )
                    st.info(f"Created new collection: {collection_id}")
                
                # Create or get vector store
                self.vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=collection_id,
                    embedding=self.embeddings
                )
                
                # Add documents to vector store
                self.vector_store.add_documents(split_docs)
                
                # Save the updated metadata to disk
                save_pdf_metadata(self.pdf_metadata)
            except Exception as e:
                st.error(f"Error initializing vector store: {str(e)}")
                raise
            
            return collection_id
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            if os.path.exists(pdf_save_path):
                # Don't delete the file on error, we want to keep it for debugging
                # Just log the error
                print(f"Error with PDF file at {pdf_save_path}: {e}")
            return None
        
    def get_retriever(self, collection_ids: List[str], use_multiquery: bool = True) -> VectorStoreRetriever:
        """
        Get a retriever for searching across specified PDF collections
        
        Args:
            collection_ids: List of collection IDs to search
            use_multiquery: Whether to use multiquery retrieval for better results
            
        Returns:
            A retriever instance
        """
        if not collection_ids:
            raise ValueError("No collections specified for retrieval")
        
        # Check if Qdrant client is available
        if qdrant_client is None:
            raise ValueError("Qdrant client is not initialized")
            
        # Get list of existing collections
        existing_collections = [c.name for c in qdrant_client.get_collections().collections]
        
        # Filter to only include collections that exist
        valid_collection_ids = [cid for cid in collection_ids if cid in existing_collections]
        
        if not valid_collection_ids:
            st.error(f"None of the specified collections exist: {collection_ids}")
            raise ValueError(f"Collections not found: {collection_ids}")
        
        # If any collections were filtered out, show a warning
        if len(valid_collection_ids) < len(collection_ids):
            missing = set(collection_ids) - set(valid_collection_ids)
            st.warning(f"Some collections were not found and will be skipped: {missing}")
        
        # If single collection, use it directly
        if len(valid_collection_ids) == 1:
            collection_id = valid_collection_ids[0]
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_id,
                embedding=self.embeddings
            )
            base_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        else:
            # For multiple collections, use a combined approach
            # First, initialize all vector stores
            vector_stores = []
            for collection_id in valid_collection_ids:
                vs = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=collection_id,
                    embedding=self.embeddings
                )
                vector_stores.append(vs)
            
            # Create a combined retriever function
            def combined_retrieval(query):
                all_docs = []
                for vs in vector_stores:
                    docs = vs.similarity_search(query, k=3)
                    all_docs.extend(docs)
                # Sort by relevance score if available
                return all_docs
            
            # Instead of creating a custom subclass of VectorStoreRetriever, which has
            # complex Pydantic validation requirements, let's create a simpler approach
            # by using LangChain's BaseRetriever directly
            from langchain_core.retrievers import BaseRetriever
            
            # Create a custom retriever that directly inherits from BaseRetriever
            class CombinedRetriever(BaseRetriever):
                """Custom retriever that combines documents from multiple collections"""
                
                def __init__(self, retrieval_function):
                    """Initialize with a retrieval function"""
                    # Call parent constructor first
                    super().__init__()
                    # Store the retrieval function as a protected attribute
                    self._retrieval_func = retrieval_function
                
                def _get_relevant_documents(self, query, *, run_manager=None):
                    """Implementation of abstract method from BaseRetriever"""
                    # Call the stored function to get documents
                    return self._retrieval_func(query)
            
            base_retriever = CombinedRetriever(combined_retrieval)
        
        # If multiquery is enabled, enhance the retriever
        if use_multiquery:
            try:
                # Get LLM for query generation
                llm = self._get_llm()
                
                # Create a multiquery retriever with the base retriever
                multiquery_prompt = PromptTemplate(
                    input_variables=["question"],
                    template="""You are an AI language model assistant. Your task is to generate multiple search queries related to a given question to help retrieve relevant documents. 
                    These queries should help find information to answer the original question.
                    
                    Generate 3 different versions of the given question to retrieve diverse and relevant information.
                    
                    Original question: {question}
                    
                    Variation 1:
                    Variation 2:
                    Variation 3:"""
                )
                
                # Check if LLM is the ErrorHandlingLLM and handle accordingly
                if getattr(llm, "_llm_type", "") == "error_handling_llm":
                    st.warning("Could not initialize an LLM for improved retrieval. Using basic retrieval instead.")
                    return base_retriever
                
                retriever = MultiQueryRetriever.from_llm(
                    retriever=base_retriever,
                    llm=llm,
                    prompt=multiquery_prompt
                )
                return retriever
            except Exception as e:
                st.warning(f"Error setting up advanced retrieval: {str(e)}. Using basic retrieval instead.")
                return base_retriever
        
        return base_retriever
    
    def _convert_pdf_to_images(self, pdf_path: str, collection_id: str):
        """
        Convert a PDF to a set of page images and store them in memory
        
        Args:
            pdf_path: Path to the PDF file
            collection_id: Collection ID to associate with the images
        """
        try:
            # Create the dictionary for this collection if it doesn't exist
            if collection_id not in self.pdf_page_images:
                self.pdf_page_images[collection_id] = {}
                
            # Convert PDF pages to images
            import io
            import base64
            from pdf2image import convert_from_path
            
            # Convert each page to an image
            print(f"Converting PDF at {pdf_path} to images...")
            images = convert_from_path(pdf_path, dpi=150)
            
            # Store each page image as a base64-encoded string
            for i, image in enumerate(images):
                # Page numbers are 1-indexed
                page_num = i + 1
                
                # Convert PIL image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Store in collection dictionary
                self.pdf_page_images[collection_id][page_num] = img_str
                print(f"Saved image for page {page_num}, size: {len(img_str)} bytes")
                
        except Exception as e:
            print(f"Error converting PDF to images: {str(e)}")
            # Initialize with empty dict if conversion failed
            if collection_id not in self.pdf_page_images:
                self.pdf_page_images[collection_id] = {}
                
    def _convert_specific_pdf_page(self, pdf_path: str, collection_id: str, page_number: int) -> str:
        """
        Convert a specific page of a PDF to an image and store it in memory
        
        Args:
            pdf_path: Path to the PDF file
            collection_id: Collection ID to associate with the image
            page_number: Page number to convert (1-indexed)
            
        Returns:
            Base64-encoded image string or empty string if conversion failed
        """
        try:
            # Create the dictionary for this collection if it doesn't exist
            if collection_id not in self.pdf_page_images:
                self.pdf_page_images[collection_id] = {}
                
            # If page already exists in cache, return it
            if page_number in self.pdf_page_images[collection_id]:
                return self.pdf_page_images[collection_id][page_number]
                
            # Convert specific PDF page to image
            import io
            import base64
            from pdf2image import convert_from_path
            
            # Convert only the specific page (first_page and last_page are 1-indexed)
            print(f"Converting page {page_number} from PDF at {pdf_path}...")
            images = convert_from_path(
                pdf_path, 
                dpi=150, 
                first_page=page_number, 
                last_page=page_number
            )
            
            if not images:
                print(f"No image generated for page {page_number}")
                return ""
                
            # Convert PIL image to base64
            image = images[0]  # Should only have one image
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Store in collection dictionary
            self.pdf_page_images[collection_id][page_number] = img_str
            print(f"Saved image for page {page_number}, size: {len(img_str)} bytes")
            
            return img_str
                
        except Exception as e:
            print(f"Error converting PDF page to image: {str(e)}")
            return ""
    
    def get_page_image(self, collection_id: str, page_number: int) -> str:
        """
        Get the base64-encoded image for a specific page
        
        Args:
            collection_id: The collection ID
            page_number: The page number (1-indexed)
            
        Returns:
            Base64-encoded image or empty string if not found
        """
        # Debug print for image retrieval attempt
        print(f"DEBUG: Attempting to get image for {collection_id}, page {page_number}")
        
        # First check if the image is in the cache
        if collection_id in self.pdf_page_images and page_number in self.pdf_page_images[collection_id]:
            image_data = self.pdf_page_images[collection_id][page_number]
            print(f"DEBUG: Found image in cache, length: {len(image_data)}")
            return image_data
        
        # If not in cache, try to generate it on demand
        try:
            pdf_metadata = self.get_pdf_metadata(collection_id)
            if pdf_metadata and "path" in pdf_metadata:
                pdf_path = pdf_metadata["path"]
                if os.path.exists(pdf_path):
                    print(f"DEBUG: Found PDF at {pdf_path}, generating page image...")
                    # Convert specific page to image
                    image_data = self._convert_specific_pdf_page(pdf_path, collection_id, page_number)
                    if image_data:
                        return image_data
                    else:
                        print(f"DEBUG: Failed to generate image for page {page_number}")
                else:
                    print(f"DEBUG: PDF path {pdf_path} does not exist")
            else:
                print(f"DEBUG: No valid metadata found for collection {collection_id}")
        except Exception as e:
            print(f"DEBUG: Error generating page image: {str(e)}")
        
        # If all else fails, return empty string
        return ""
    
    def get_pdf_metadata(self, collection_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific PDF
        
        Args:
            collection_id: The collection ID
            
        Returns:
            Dictionary with PDF metadata
        """
        print(f"DEBUG: get_pdf_metadata called for collection_id: {collection_id}")
        
        # Always reload metadata from disk to ensure we have the latest data
        # This helps avoid stale metadata issues across Streamlit reruns
        self.pdf_metadata = load_pdf_metadata()
        print(f"DEBUG: Loaded metadata from disk, keys: {list(self.pdf_metadata.keys())}")
        
        metadata = self.pdf_metadata.get(collection_id, {})
        
        if metadata:
            print(f"DEBUG: Found metadata for {collection_id} in loaded data")
            
            # Check if the path in metadata is valid
            if "path" in metadata and not os.path.exists(metadata["path"]):
                print(f"DEBUG: Path in metadata {metadata['path']} does not exist, checking alternates")
                
                # Try alternative path
                alt_path = os.path.join("uploaded_pdfs", f"{collection_id}.pdf")
                if os.path.exists(alt_path):
                    print(f"DEBUG: Found file at alternative path: {alt_path}")
                    metadata["path"] = alt_path
                    metadata["abs_path"] = os.path.abspath(alt_path)
                    # Update metadata
                    self.pdf_metadata[collection_id] = metadata
                    save_pdf_metadata(self.pdf_metadata)
        else:
            print(f"DEBUG: No metadata found for collection {collection_id}")
            
            # If we can't find metadata, try to check if the file exists directly in uploaded_pdfs
            potential_path = os.path.join("uploaded_pdfs", f"{collection_id}.pdf")
            if os.path.exists(potential_path):
                print(f"DEBUG: Found PDF file at {potential_path} but no metadata")
                # Create basic metadata
                basic_metadata = {
                    "filename": f"{collection_id}.pdf",
                    "collection_id": collection_id,
                    "path": potential_path,
                    "abs_path": os.path.abspath(potential_path),
                    "pages": 0  # We don't know the page count yet
                }
                # Add to cache and save
                self.pdf_metadata[collection_id] = basic_metadata
                save_pdf_metadata(self.pdf_metadata)
                return basic_metadata
            
        # Return the metadata or an empty dict if not found
        return metadata
    
    def extract_page_text(self, collection_id: str, page_number: int) -> str:
        """
        Extract text from a specific page for highlighting
        
        Args:
            collection_id: The collection ID
            page_number: The page number (1-indexed)
            
        Returns:
            Page text
        """
        pdf_metadata = self.get_pdf_metadata(collection_id)
        if not pdf_metadata:
            return ""
        
        pdf_path = pdf_metadata.get("path")
        if not pdf_path or not os.path.exists(pdf_path):
            return ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if 1 <= page_number <= len(pdf.pages):
                    return pdf.pages[page_number - 1].extract_text() or ""
                return ""
        except Exception as e:
            st.error(f"Error extracting page text: {e}")
            return ""
    
    def delete_collection(self, collection_id: str) -> bool:
        """
        Delete a PDF collection from the vector database
        
        Args:
            collection_id: The collection ID to delete
            
        Returns:
            Success status
        """
        if not collection_id or not collection_id.startswith("pdf_"):
            return False
            
        try:
            # Delete the collection from Qdrant
            if qdrant_client:
                qdrant_client.delete_collection(collection_name=collection_id)
            
            # Remove from local cache
            if collection_id in self.pdf_metadata:
                # Clean up temp file if it exists
                pdf_path = self.pdf_metadata[collection_id].get("path")
                if pdf_path and os.path.exists(pdf_path):
                    try:
                        os.unlink(pdf_path)
                    except Exception:
                        pass  # Ignore errors on temp file cleanup
                
                # Remove from metadata cache
                del self.pdf_metadata[collection_id]
            
            # Clean up page images cache
            if collection_id in self.pdf_page_images:
                del self.pdf_page_images[collection_id]
            
            # Save the updated metadata to disk    
            save_pdf_metadata(self.pdf_metadata)
                
            return True
            
        except Exception as e:
            st.error(f"Error deleting collection: {e}")
            return False

    def _get_llm(self):
        """Get LLM based on configured provider and model"""
        # Create a LangChain compatible LLM wrapper
        from langchain_core.language_models.llms import LLM
        from typing import Optional, Any, ClassVar
        
        class CustomLLM(LLM):
            """Custom LLM wrapper for model_clients"""
            client: Optional[Any] = None  # Initialize client as None with proper type annotation
            
            def __init__(self, client=None):
                super().__init__()
                if client:
                    self.client = client
                
            def _call(self, prompt, **kwargs):
                if not self.client:
                    return "Error: Model client not initialized properly. Unable to process query."
                try:
                    messages = [{"role": "user", "content": prompt}]
                    return self.client.generate_completion(messages=messages)
                except Exception as e:
                    return f"Error generating completion: {str(e)}"
                
            @property
            def _llm_type(self):
                return "custom_llm"
        
        # Create fallback error handler class
        class ErrorHandlingLLM(LLM):
            """Error handler LLM for when client initialization fails"""
            error_message: str = "Initialization failed"  # Default error message with type annotation
            
            def __init__(self, error_message="Initialization failed"):
                super().__init__()
                self.error_message = error_message
            
            def _call(self, prompt, **kwargs):
                return f"Error: {self.error_message}. Unable to process query."
                
            @property
            def _llm_type(self):
                return "error_handling_llm"
        
        # Try to get an appropriate client with robust error handling
        try:
            client = model_clients.get_model_client(
                provider=self.model_provider,
                model_name=self.model_name
            )
            return CustomLLM(client)
        except Exception as e:
            st.error(f"Error initializing model client: {str(e)}")
            
            # Try OpenAI as fallback
            try:
                fallback_client = model_clients.get_model_client("openai", "gpt-4o")
                st.warning("Using fallback OpenAI model. Check your API settings.")
                return CustomLLM(fallback_client)
            except Exception as e2:
                error_msg = f"Could not initialize any model client. Please check your API keys in the sidebar settings. Error: {str(e2)}"
                st.error(error_msg)
                return ErrorHandlingLLM(error_msg)
                
    def create_interactive_pdf_viewer(self, collection_id, metadata, pdf_processor=None, query_terms=None, highlight_page=None):
        """
        Create an interactive PDF viewer with text highlighting capability
        
        Args:
            collection_id: The collection ID for the PDF
            metadata: PDF metadata dictionary
            pdf_processor: Optional, for backward compatibility (not used as self is available)
            query_terms: Optional list of terms to highlight
            highlight_page: Optional specific page to highlight
            
        Returns:
            HTML for an interactive PDF viewer with highlighting capability
        """
        # First, check what files actually exist in the uploaded_pdfs directory
        import glob
        actual_pdfs = glob.glob("uploaded_pdfs/*.pdf")
        print(f"DEBUG: Actually available PDFs when creating viewer: {actual_pdfs}")
        
        if not metadata:
            metadata = {}
            print(f"DEBUG: No metadata provided, creating empty metadata")
            
        print(f"DEBUG: Creating PDF viewer for collection_id={collection_id}")
        if metadata:
            print(f"DEBUG: Metadata keys: {list(metadata.keys())}")
        
        # First check for direct matches based on collection ID (most reliable)
        matching_pdfs = [pdf for pdf in actual_pdfs if collection_id in pdf]
        if matching_pdfs:
            pdf_path = matching_pdfs[0]
            file_exists = True
            print(f"DEBUG: Using exact match PDF for {collection_id}: {pdf_path}")
            metadata["path"] = pdf_path
            
            # Update stored metadata with the correct path
            if hasattr(self, 'pdf_metadata'):
                if collection_id in self.pdf_metadata:
                    self.pdf_metadata[collection_id]["path"] = pdf_path
                    save_pdf_metadata(self.pdf_metadata)
                    print(f"DEBUG: Updated metadata with path: {pdf_path}")
        # If no direct match, check path from metadata            
        else:
            pdf_path = metadata.get("path")
            print(f"DEBUG: PDF path from metadata: {pdf_path}")
            
            # Check if we're missing the path or the file doesn't exist
            file_exists = False
            if not pdf_path:
                print(f"DEBUG: No path in metadata for collection {collection_id}")
            else:
                # Check if the file exists at the specified path
                if os.path.exists(pdf_path):
                    file_exists = True
                    print(f"DEBUG: Found PDF at original path: {pdf_path}")
            
            # If no exact collection match but we have PDFs - use the first available one
            if not file_exists and len(actual_pdfs) > 0:
                pdf_path = actual_pdfs[0]
                file_exists = True
                print(f"DEBUG: Using first available PDF as fallback: {pdf_path}")
                
                # Update the metadata
                metadata["path"] = pdf_path
                if hasattr(self, 'pdf_metadata') and collection_id in self.pdf_metadata:
                    self.pdf_metadata[collection_id]["path"] = pdf_path
                    save_pdf_metadata(self.pdf_metadata)
                    print(f"DEBUG: Updated metadata with fallback path: {pdf_path}")
            # If still not found, try alternate paths
            elif not file_exists:
                # Try possible locations
                possible_paths = [
                    # Common relative paths
                    os.path.join("uploaded_pdfs", f"{collection_id}.pdf"),
                    os.path.join(".", "uploaded_pdfs", f"{collection_id}.pdf"),
                    # Try with original filename if available
                    os.path.join("uploaded_pdfs", metadata.get("filename", "")),
                    # Try normalized paths
                    os.path.normpath(os.path.join("uploaded_pdfs", f"{collection_id}.pdf")),
                    os.path.normpath(pdf_path) if pdf_path else None
                ]
                
                # Filter out None or empty paths
                possible_paths = [p for p in possible_paths if p]
                # Remove duplicates while preserving order
                seen = set()
                possible_paths = [p for p in possible_paths if not (p in seen or seen.add(p))]
                
                print(f"DEBUG: Trying these possible paths: {possible_paths}")
                
                # Try each path
                for alt_path in possible_paths:
                    if os.path.exists(alt_path):
                        print(f"DEBUG: Found PDF at alternative path: {alt_path}")
                        # Update the metadata with the correct path
                        metadata["path"] = alt_path
                        pdf_path = alt_path
                        file_exists = True
                        # Also update metadata on disk for future use
                        if hasattr(self, 'pdf_metadata'):
                            if collection_id in self.pdf_metadata:
                                self.pdf_metadata[collection_id]["path"] = alt_path
                                save_pdf_metadata(self.pdf_metadata)
                                print(f"DEBUG: Updated stored metadata for {collection_id} with new path")
                        break
                        
                if not file_exists:
                    print(f"DEBUG: PDF file not found at any path for collection {collection_id}")
                    print(f"DEBUG: Tried paths: {', '.join(possible_paths)}")
        
        # If the file doesn't exist anywhere, show a fallback viewer with just the text content
        if not file_exists:
            print(f"DEBUG: PDF file not found anywhere for collection {collection_id}")
            
            # Get content from Qdrant so we can at least show the text
            try:
                # Let's create a fallback view with the text content from the vector store
                client = qdrant_client
                if not client:
                    return "<p>PDF file not found and vector database is not available.</p>"
                
                # Query the collection to get documents
                query_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="collection_id",
                            match=qdrant_models.MatchValue(value=collection_id)
                        )
                    ]
                )
                
                # If we have a highlight page specified, restrict to that page
                if highlight_page:
                    query_filter.must.append(
                        qdrant_models.FieldCondition(
                            key="page_number",
                            match=qdrant_models.MatchValue(value=highlight_page)
                        )
                    )
                    
                # Get the documents from the vector store
                search_result = client.scroll(
                    collection_name=collection_id,
                    scroll_filter=query_filter,
                    limit=100,
                    with_payload=True
                )[0]
                
                if not search_result:
                    return "<p>Content not found in the database for this document.</p>"
                    
                # Extract text content and organize by page
                pages_content = {}
                for hit in search_result:
                    payload = hit.payload
                    page_num = payload.get("page_number", 0)
                    text = payload.get("text", "")
                    
                    if page_num not in pages_content:
                        pages_content[page_num] = ""
                        
                    pages_content[page_num] += text + "\n\n"
                
                # Build HTML for the fallback view
                html = f"""
                <div style="border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px;">
                    <h3>PDF File Not Found</h3>
                    <p>The original PDF file is no longer available, but text content has been retrieved from the database.</p>
                """
                
                # Add content from the relevant page if highlight_page is specified
                if highlight_page and highlight_page in pages_content:
                    html += f"<h4>Content from Page {highlight_page}:</h4>"
                    
                    # Get the page content
                    page_text = pages_content[highlight_page]
                    
                    # Highlight terms if provided
                    if query_terms and len(query_terms) > 0:
                        highlight_colors = ['#FFFF00', '#90EE90', '#ADD8E6', '#FFB6C1', '#D8BFD8']
                        for i, term in enumerate(query_terms):
                            color = highlight_colors[i % len(highlight_colors)]
                            page_text = page_text.replace(
                                term, 
                                f'<span style="background-color: {color};">{term}</span>'
                            )
                    
                    # Replace newlines with HTML line breaks for display
                    page_text = page_text.replace("\n", "<br>")
                    html += f'<div style="background-color: #f9f9f9; padding: 15px; border-radius: 3px;">{page_text}</div>'
                
                # If no specific page or page not found, show all pages
                else:
                    for page_num in sorted(pages_content.keys()):
                        html += f"<h4>Page {page_num}:</h4>"
                        page_text = pages_content[page_num].replace("\n", "<br>")
                        
                        # Highlight terms if provided
                        if query_terms and len(query_terms) > 0:
                            highlight_colors = ['#FFFF00', '#90EE90', '#ADD8E6', '#FFB6C1', '#D8BFD8']
                            for i, term in enumerate(query_terms):
                                color = highlight_colors[i % len(highlight_colors)]
                                page_text = page_text.replace(
                                    term, 
                                    f'<span style="background-color: {color};">{term}</span>'
                                )
                        
                        html += f'<div style="background-color: #f9f9f9; padding: 15px; border-radius: 3px; margin-bottom: 10px;">{page_text}</div>'
                
                html += "</div>"
                return html
                
            except Exception as e:
                print(f"DEBUG: Error creating fallback view: {str(e)}")
                return f"<p>PDF file not found and there was an error retrieving text content: {str(e)}</p>"
        
        try:
            # Extract text content from PDF for highlighting
            pdf_text = {}
            with open(pdf_path, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                total_pages = len(pdf_reader.pages)
                
                # Extract text from each page
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    # Store with 1-indexed page numbers for consistency with other functions
                    pdf_text[page_num + 1] = page.extract_text()
            
            # Convert PDF to base64 for embedding
            with open(pdf_path, "rb") as f:
                pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Prepare highlight data if query terms are provided
            highlight_data = {}
            # Define list of highlight colors for different terms
            highlight_colors = ['#FFFF00', '#90EE90', '#ADD8E6', '#FFB6C1', '#D8BFD8']
            
            if query_terms and len(query_terms) > 0:  # Check if query_terms is not empty
                # Process each page for highlights
                for page_num, text in pdf_text.items():
                    if highlight_page and page_num != highlight_page:
                        continue
                    
                    # Find term occurrences in the page text
                    highlights = []
                    text_lower = text.lower()
                    for i, term in enumerate(query_terms):
                        term_lower = term.lower()
                        start_pos = 0
                        # Get color for current term (cycle through colors if more terms than colors)
                        color = highlight_colors[i % len(highlight_colors)]
                        
                        while start_pos < len(text_lower):
                            pos = text_lower.find(term_lower, start_pos)
                            if pos == -1:
                                break
                                
                            # Get a window around the term
                            context_start = max(0, pos - 100)
                            context_end = min(len(text), pos + len(term) + 100)
                            
                            # Get the actual term with original casing
                            term_actual = text[pos:pos+len(term)]
                            
                            # Store the highlight information with color
                            highlights.append({
                                "term": term_actual,
                                "position": pos,
                                "context": text[context_start:context_end],
                                "color": color
                            })
                            
                            # Move past this occurrence
                            start_pos = pos + len(term)
                    
                    if highlights:
                        highlight_data[page_num] = highlights
            
            # Create a simplified PDF viewer using a direct embed approach
            # This is more reliable across browsers
            
            # Add page navigation for the PDF
            current_page = highlight_page if highlight_page else 1
            
            # Create HTML for the improved PDF viewer with simplified highlighting
            viewer_html = f"""
            <div class="pdf-viewer-container" style="width: 100%; margin-top: 15px;">
                <div class="pdf-toolbar" style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background: #f5f5f5; border: 1px solid #ddd; border-bottom: none; border-radius: 4px 4px 0 0;">
                    <div class="pdf-info">
                        <span style="font-weight: bold;">Page {current_page} of {len(pdf_reader.pages)}</span>
                    </div>
                    <div class="pdf-highlight-controls">
                        <div id="highlight-legend" style="font-size: 12px; display: inline-block;"></div>
                    </div>
                </div>
                
                <div style="position: relative; width: 100%; height: 800px; border: 1px solid #ddd; border-radius: 0 0 4px 4px; overflow: hidden;">
                    <iframe 
                        src="data:application/pdf;base64,{pdf_base64}#{current_page}" 
                        width="100%" 
                        height="100%"
                        style="border: none;">
                    </iframe>
                </div>
                <div style="margin-top: 10px; text-align: center;">
                    <small>Viewing: {os.path.basename(pdf_path)} - Page {current_page}</small>
                </div>
                <div style="margin-top: 10px;">
                    <a href="data:application/pdf;base64,{pdf_base64}" 
                       download="{os.path.basename(pdf_path)}"
                       style="text-decoration: none; color: #4361ee; font-size: 14px;">
                       📥 Download PDF
                    </a>
                </div>
            </div>
            
            <script>
                // Create a legend for different highlight terms
                (function createHighlightLegend() {{
                    const legend = document.getElementById('highlight-legend');
                    if (!legend) return;
                    
                    // Define highlight colors
                    const highlight_colors = {json.dumps(highlight_colors)};
                    const query_terms = {json.dumps(query_terms) if query_terms else "[]"};
                    
                    // If we have terms to display
                    if (query_terms && query_terms.length > 0) {{
                        const legendSpan = document.createElement('span');
                        legendSpan.innerHTML = '<strong>Search Terms:</strong> ';
                        legend.appendChild(legendSpan);
                        
                        // Create legend items
                        query_terms.forEach((term, index) => {{
                            const color = highlight_colors[index % highlight_colors.length];
                            
                            const termSpan = document.createElement('span');
                            termSpan.style.display = 'inline-block';
                            termSpan.style.margin = '0 5px';
                            termSpan.style.padding = '2px 5px';
                            termSpan.style.backgroundColor = color;
                            termSpan.style.borderRadius = '3px';
                            termSpan.textContent = term;
                            
                            legend.appendChild(termSpan);
                        }});
                    }}
                }})();
            </script>
            """
            
            return viewer_html
            
        except Exception as e:
            st.error(f"Error creating PDF viewer: {str(e)}")
            return f"<p>Error creating PDF viewer: {str(e)}</p>"


def save_pdf_metadata(metadata_dict):
    """Save PDF metadata to disk"""
    try:
        # Convert any non-serializable data to serializable format
        clean_metadata = {}
        
        # First, check what files actually exist in the uploaded_pdfs directory
        import glob
        actual_pdfs = glob.glob("uploaded_pdfs/*.pdf")
        print(f"DEBUG: Actually available PDFs when saving metadata: {actual_pdfs}")
        
        for cid, metadata in metadata_dict.items():
            # Create a copy without any non-serializable data
            clean_meta = metadata.copy()
            
            # Always look for exact PDF match first based on collection ID
            matching_pdfs = [pdf for pdf in actual_pdfs if cid in pdf]
            if matching_pdfs:
                clean_meta["path"] = matching_pdfs[0]
                print(f"DEBUG: Found exact match for {cid}: {matching_pdfs[0]}")
            # Then try paths in the existing metadata
            elif "path" in clean_meta:
                # Check if file exists at path
                if not os.path.exists(clean_meta["path"]):
                    # Try to find the file with various strategies
                    if os.path.exists(f"uploaded_pdfs/{cid}.pdf"):
                        clean_meta["path"] = f"uploaded_pdfs/{cid}.pdf"
                    elif os.path.exists(os.path.join("uploaded_pdfs", clean_meta.get("filename", ""))):
                        clean_meta["path"] = os.path.join("uploaded_pdfs", clean_meta["filename"])
                    # Last resort: if there's only one PDF, use it
                    elif len(actual_pdfs) == 1:
                        clean_meta["path"] = actual_pdfs[0]
                        print(f"DEBUG: Using only available PDF for {cid}: {actual_pdfs[0]}")
                
                # Make sure path uses a consistent format - simple relative path
                clean_meta["path"] = clean_meta["path"].replace("./", "")
                if clean_meta["path"].startswith("/repo/"):
                    clean_meta["path"] = clean_meta["path"].replace("/repo/", "")
                
                print(f"DEBUG: Saving metadata with path: {clean_meta['path']} for collection {cid}")
            # If path is missing, reconstruct it (for backward compatibility)
            elif not "path" in clean_meta:
                expected_path = f"uploaded_pdfs/{cid}.pdf"
                if os.path.exists(expected_path):
                    clean_meta["path"] = expected_path
                # If no exact match exists, use generic path
                else:
                    clean_meta["path"] = f"uploaded_pdfs/{cid}.pdf"
            
            # Double check that our path exists
            if "path" in clean_meta and not os.path.exists(clean_meta["path"]):
                print(f"WARNING: Metadata references non-existent path: {clean_meta['path']}")
                
            clean_metadata[cid] = clean_meta
        
        with open(PDF_METADATA_FILE, "w") as f:
            json.dump(clean_metadata, f)
        return True
    except Exception as e:
        st.error(f"Error saving PDF metadata: {e}")
        return False

def load_pdf_metadata():
    """Load PDF metadata from disk"""
    if not os.path.exists(PDF_METADATA_FILE):
        return {}
    
    try:
        with open(PDF_METADATA_FILE, "r") as f:
            metadata = json.load(f)
            
        # Ensure all entries have path information
        for cid, meta in metadata.items():
            # If path is missing, add a default path to uploaded_pdfs directory
            if "path" not in meta:
                default_path = f"uploaded_pdfs/{cid}.pdf"
                meta["path"] = default_path
                print(f"Adding default path for {cid}: {default_path}")
            
            # Clean up path format - standardize to simple relative paths without ./ prefix
            if meta["path"].startswith("./"):
                meta["path"] = meta["path"].replace("./", "")
                
            if meta["path"].startswith("/repo/"):
                meta["path"] = meta["path"].replace("/repo/", "")
                
            # Verify if the file exists at the specified path
            if not os.path.exists(meta["path"]):
                # Try various alternative paths
                alt_paths = [
                    os.path.join("uploaded_pdfs", f"{cid}.pdf"),
                    os.path.join(".", "uploaded_pdfs", f"{cid}.pdf"),
                    os.path.join("uploaded_pdfs", meta.get("filename", "")),
                    os.path.normpath(meta["path"]),
                    meta["path"].replace("./", "")
                ]
                
                # Find the first path that exists
                found = False
                for alt_path in alt_paths:
                    if alt_path and os.path.exists(alt_path):
                        print(f"Updating path for {cid} from {meta['path']} to {alt_path}")
                        meta["path"] = alt_path
                        found = True
                        break
                
                if not found:
                    paths_tried = ", ".join([p for p in alt_paths if p])
                    print(f"Warning: PDF file not found at {meta['path']} or alternatives tried: {paths_tried}")
                    
        return metadata
    except Exception as e:
        st.error(f"Error loading PDF metadata: {e}")
        return {}

def find_collections():
    """Find all available PDF collections in the Qdrant database"""
    if not qdrant_client:
        return []
    
    try:
        collections = qdrant_client.get_collections().collections
        return [c.name for c in collections if c.name.startswith("pdf_")]
    except Exception as e:
        st.error(f"Error listing collections: {e}")
        return []


def highlight_text_in_context(text, query_terms, context_window=100):
    """
    Highlight occurrences of query terms in text with surrounding context
    
    Args:
        text: The full page text
        query_terms: List of terms to highlight
        context_window: Number of characters before and after to include
        
    Returns:
        List of highlighted text snippets with context
    """
    if not text or not query_terms:
        return []
    
    # Define color palette for different terms (matching the colors in create_interactive_pdf_viewer)
    highlight_colors = ['#FFFF00', '#90EE90', '#ADD8E6', '#FFB6C1', '#D8BFD8']
    
    snippets = []
    text_lower = text.lower()
    
    for i, term in enumerate(query_terms):
        term_lower = term.lower()
        start_pos = 0
        
        # Get color for this term (cycle through colors if more terms than colors)
        highlight_color = highlight_colors[i % len(highlight_colors)]
        
        while True:
            pos = text_lower.find(term_lower, start_pos)
            if pos == -1:
                break
                
            # Calculate context window
            snippet_start = max(0, pos - context_window)
            snippet_end = min(len(text), pos + len(term) + context_window)
            
            # Extract context
            context_before = text[snippet_start:pos]
            highlighted_term = text[pos:pos+len(term)]
            context_after = text[pos+len(term):snippet_end]
            
            # Create snippet with HTML highlighting
            snippet = {
                "before": context_before,
                "highlight": highlighted_term,
                "after": context_after,
                "page": None,  # Will be set by caller
                "color": highlight_color  # Add color information
            }
            
            snippets.append(snippet)
            start_pos = pos + len(term)
    
    return snippets


# This function has been moved to the PDFProcessor class as a method
# Keeping this stub for backward compatibility with older code
def create_interactive_pdf_viewer(collection_id, metadata, pdf_processor=None, query_terms=None, highlight_page=None):
    """
    DEPRECATED: This function is now available as a method on the PDFProcessor class.
    
    Args:
        collection_id: The collection ID for the PDF
        metadata: PDF metadata dictionary
        pdf_processor: An instance of PDFProcessor (required)
        query_terms: Optional list of terms to highlight
        highlight_page: Optional specific page to highlight
        
    Returns:
        HTML for an interactive PDF viewer with highlighting capability
    """
    if pdf_processor is None:
        return "<p>Error: PDF processor instance is required.</p>"
        
    return pdf_processor.create_interactive_pdf_viewer(
        collection_id=collection_id,
        metadata=metadata,
        query_terms=query_terms,
        highlight_page=highlight_page
    )


def extract_query_terms(query):
    """
    Extract key terms from the query for highlighting
    
    Args:
        query: The user query
        
    Returns:
        List of key terms
    """
    # Remove common words and punctuation
    import re
    import string
    
    # Remove punctuation and convert to lowercase
    query = query.lower()
    query = query.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words
    words = query.split()
    
    # Remove common stop words
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
        'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
        'to', 'from', 'in', 'on', 'by', 'about', 'like', 'with', 'after', 'between',
        'at', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'but',
        'should', 'can', 'could', 'would', 'will', 'shall', 'not', 'no', 'nor',
        'are', 'were', 'was', 'am', 'i', 'we', 'they', 'he', 'she', 'it'
    }
    
    # Filter out stop words and ensure terms are at least 3 characters
    terms = [word for word in words if word not in stop_words and len(word) >= 3]
    
    # Prioritize longer terms
    terms.sort(key=len, reverse=True)
    
    # Take top 5 terms
    return terms[:5]