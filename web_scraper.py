"""
Web Scraper Module for DB Insights Explorer

This module handles web scraping, text processing and vector storage for website content.
It allows the user to scrape websites, store the vectorized content, and query it.
"""

# Version 1.2.0 - Enhanced with Selenium support and vectorization improvements

import os
import time
import json
import logging
import re
import random
import urllib.parse
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import requests
import trafilatura
import html2text
from bs4 import BeautifulSoup
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

# Configure OpenAI for embeddings
import openai
import tiktoken

# Import Selenium for advanced scraping
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Import Apify Crawler for advanced website crawling
try:
    from apify_crawler import ApifyCrawler, get_crawl_data_as_documents
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_website(url: str, collection_name: str, use_selenium: bool = False, 
               use_premium: bool = False, premium_render_js: bool = True, premium_country: str = "",
               recursive: bool = False, max_pages: int = 10, same_domain_only: bool = True,
               metadata: Dict = None, use_apify: bool = False, max_crawl_depth: int = 5) -> Dict:
    """
    Main function to scrape a website, chunk the content, and store it in ChromaDB.
    
    Args:
        url: URL of the website to scrape
        collection_name: Name of the ChromaDB collection to store the content
        use_selenium: Whether to use Selenium for scraping (useful for JS-heavy sites)
        use_premium: Whether to use premium ScrapingBee service
        premium_render_js: Whether to render JavaScript with ScrapingBee
        premium_country: Country code for ScrapingBee geolocation (e.g., "us")
        recursive: Whether to recursively crawl the website following links
        max_pages: Maximum number of pages to crawl when recursive is True
        same_domain_only: When recursive is True, only follow links to the same domain
        metadata: Additional metadata to store with the content
        use_apify: Whether to use Apify Website Content Crawler for advanced crawling
        max_crawl_depth: Maximum depth for Apify crawling (when use_apify is True)
        
    Returns:
        Dictionary with results of the scraping operation
    """
    try:
        # Initialize the ChromaDB client
        client = get_chroma_client()
        if not client:
            return {"success": False, "error": "Failed to initialize ChromaDB client"}
        
        # Get the OpenAI embedding function
        embedding_function = get_openai_embedding_function()
        if not embedding_function:
            return {"success": False, "error": "Failed to initialize OpenAI embedding function. Please check your API key."}
        
        # Sanitize collection name for ChromaDB compatibility
        sanitized_name = sanitize_collection_name(collection_name)
        logger.info(f"Using ChromaDB v0.6+ collection names")
        
        # Check if Apify should be used for advanced crawling
        if use_apify:
            if not APIFY_AVAILABLE:
                return {"success": False, "error": "Apify integration is not available. Please install the apify-client package."}
            
            # Initialize Apify crawler
            from apify_crawler import apify_crawler, get_crawl_data_as_documents
            
            if not apify_crawler.is_available():
                return {"success": False, "error": "Apify API key not found. Please add APIFY_API_KEY to your environment variables."}
            
            logger.info(f"Using Apify Website Content Crawler to crawl {url}")
            
            # Use Apify to crawl the website
            # If same_domain_only is True, the crawler will already stay on the same domain via includeUrlGlobs
            # For exclude_patterns, we'd add specific paths to avoid if needed
            exclude_patterns = []
            
            # In future versions, we could add more specific exclusion patterns here
            # based on user configuration or common patterns to ignore
            
            crawl_results = apify_crawler.crawl_website(
                url=url,
                max_crawl_pages=max_pages,
                max_crawl_depth=max_crawl_depth,
                exclude_patterns=exclude_patterns
            )
            
            if not crawl_results["success"]:
                return {"success": False, "error": f"Apify crawling failed: {crawl_results.get('error', 'Unknown error')}"}
            
            # Convert crawl results to documents
            documents = get_crawl_data_as_documents(crawl_results)
            
            if not documents:
                return {"success": False, "error": "No content extracted from Apify crawl"}
            
            # Get or create the collection
            try:
                # Try to get existing collection
                collection = client.get_collection(name=sanitized_name, embedding_function=embedding_function)
                logger.info(f"Retrieved existing collection: {sanitized_name}")
            except Exception as e:
                # Create new collection if it doesn't exist
                logger.info(f"Creating new collection: {sanitized_name}")
                collection = client.create_collection(
                    name=sanitized_name,
                    embedding_function=embedding_function,
                    metadata={"source": "apify_crawler", "timestamp": time.time()}
                )
                
            # Add documents to collection
            total_chunks_added = 0
            for i, doc in enumerate(documents):
                # Prepare metadata and ensure no None values (ChromaDB doesn't accept None values)
                chunk_metadata = {
                    "source": doc["metadata"]["url"] or "",
                    "title": doc["metadata"]["title"] or "",
                    "chunk_index": doc["metadata"]["chunk"],
                    "total_chunks": doc["metadata"]["total_chunks"],
                    "scrape_time": time.time(),
                    "crawler": "apify",
                }
                
                # Add any additional metadata provided
                if metadata:
                    # Sanitize metadata by removing None values
                    sanitized_metadata = {k: v for k, v in metadata.items() if v is not None}
                    chunk_metadata.update(sanitized_metadata)
                
                # Add the document to the collection
                collection.add(
                    ids=[f"{sanitized_name}_apify_{i}"],
                    documents=[doc["content"]],
                    metadatas=[chunk_metadata]
                )
                total_chunks_added += 1
            
            logger.info(f"Successfully added {total_chunks_added} chunks from Apify crawl to collection {sanitized_name}")
            
            # Collect URLs of crawled pages from document metadata
            urls_crawled = []
            for doc in documents:
                page_url = doc["metadata"].get("url", "")
                if page_url and page_url not in urls_crawled:
                    urls_crawled.append(page_url)
            
            # Return results
            return {
                "success": True,
                "collection_name": sanitized_name,
                "chunk_count": total_chunks_added,
                "pages_crawled": len(urls_crawled),
                "start_url": url,
                "urls_crawled": urls_crawled,
                "metadata": metadata,
                "crawler": "apify"
            }
        
        # Continue with regular crawling if not using Apify
        # Get or create the collection
        try:
            # Try to get existing collection
            collection = client.get_collection(name=sanitized_name, embedding_function=embedding_function)
            logger.info(f"Retrieved existing collection: {sanitized_name}")
        except Exception as e:
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection: {sanitized_name}")
            collection = client.create_collection(
                name=sanitized_name,
                embedding_function=embedding_function,
                metadata={"source": "web_scraper", "timestamp": time.time()}
            )
        
        # For recursive crawling, we'll need to keep track of visited URLs and links to crawl
        crawled_urls = set()
        urls_to_crawl = [url]
        total_chunks_added = 0
        chunk_offset = 0  # For ID generation
        
        # Extract the base domain for filtering links if same_domain_only is True
        base_domain = extract_domain(url) if same_domain_only else None
        
        # Keep crawling until we run out of URLs or hit the max_pages limit
        while urls_to_crawl and (not recursive or len(crawled_urls) < max_pages):
            current_url = urls_to_crawl.pop(0)
            
            # Skip if we've already crawled this URL
            if current_url in crawled_urls:
                continue
                
            try:
                # Scrape the current URL
                logger.info(f"Scraping content from {current_url}")
                
                # Use the appropriate scraping method based on settings
                if use_premium:
                    # Pass premium settings through to the get_website_text_content function
                    text_content = get_website_text_content(
                        current_url, 
                        use_selenium=False, 
                        use_premium=True,
                        premium_render_js=premium_render_js, 
                        premium_country=premium_country
                    )
                else:
                    text_content = get_website_text_content(current_url, use_selenium=use_selenium)
                
                if not text_content or not text_content.strip():
                    logger.warning(f"No content extracted from {current_url}, skipping")
                    crawled_urls.add(current_url)  # Mark as crawled even if no content
                    continue
                    
                logger.info(f"Successfully extracted {len(text_content)} characters from {current_url}")
                
                # If recursive, extract links for further crawling
                if recursive:
                    # Get links from the current page
                    try:
                        links = extract_links(text_content, current_url, base_domain)
                        # Add new links to the queue if they haven't been crawled yet
                        for link in links:
                            if link not in crawled_urls and link not in urls_to_crawl:
                                urls_to_crawl.append(link)
                        logger.info(f"Found {len(links)} new links to crawl from {current_url}")
                    except Exception as link_error:
                        logger.error(f"Error extracting links from {current_url}: {str(link_error)}")
                
                # Chunk the text for storage
                chunks = chunk_text(text_content)
                if not chunks:
                    logger.warning(f"Failed to chunk text content from {current_url}, skipping")
                    crawled_urls.add(current_url)  # Mark as crawled even if chunking failed
                    continue
                    
                logger.info(f"Created {len(chunks)} chunks from {current_url}")
                
                # Prepare IDs, metadatas and documents for batch insertion
                ids = [f"{sanitized_name}_{i + chunk_offset}" for i in range(len(chunks))]
                
                # Create metadata for each chunk
                metadatas = []
                for i in range(len(chunks)):
                    chunk_metadata = {
                        "source": current_url,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "scrape_time": time.time(),
                        "recursive_crawl": recursive,
                    }
                    
                    # Add any additional metadata provided
                    if metadata:
                        # Sanitize metadata by removing None values
                        sanitized_metadata = {k: v if v is not None else "" for k, v in metadata.items()}
                        chunk_metadata.update(sanitized_metadata)
                        
                    metadatas.append(chunk_metadata)
                
                # Add the chunks to the collection
                collection.add(
                    ids=ids,
                    documents=chunks,
                    metadatas=metadatas
                )
                
                logger.info(f"Successfully added {len(chunks)} chunks from {current_url} to collection {sanitized_name}")
                
                # Update the total chunks counter and offset
                total_chunks_added += len(chunks)
                chunk_offset += len(chunks)
                
                # Mark this URL as crawled
                crawled_urls.add(current_url)
                
            except Exception as page_error:
                logger.error(f"Error scraping content from {current_url}: {str(page_error)}")
                # Mark as crawled even if there was an error
                crawled_urls.add(current_url)
                continue
        
        # Return the results
        if recursive:
            return {
                "success": True,
                "collection_name": sanitized_name,
                "chunk_count": total_chunks_added,
                "pages_crawled": len(crawled_urls),
                "urls_crawled": list(crawled_urls),
                "start_url": url,
                "metadata": metadata
            }
        else:
            # Return original format for single-page mode to maintain compatibility
            return {
                "success": True,
                "collection_name": sanitized_name,
                "chunk_count": total_chunks_added,
                "url": url,
                "metadata": metadata
            }
            
    except Exception as e:
        logger.error(f"Error in scrape_website: {str(e)}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}
        
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of specified size
    
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
    text = text.replace("\n", " ").replace("\r", " ")
    
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If adding this sentence would exceed the chunk size, save the current chunk
        if current_size + sentence_len + 1 > chunk_size and current_chunk:  # +1 for the space
            chunks.append(" ".join(current_chunk))
            
            # If overlap is desired, keep some sentences for the next chunk
            if overlap > 0:
                # Calculate how many sentences to keep for overlap
                overlap_size = 0
                sentences_to_keep = []
                
                # Start from the end of the current chunk
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_size += len(current_chunk[i])
                    if overlap_size > overlap:
                        break
                    sentences_to_keep.insert(0, current_chunk[i])
                
                # Start the new chunk with overlapping sentences
                current_chunk = sentences_to_keep
                current_size = sum(len(s) for s in sentences_to_keep)
            else:
                # No overlap, start with an empty chunk
                current_chunk = []
                current_size = 0
        
        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_size += sentence_len + 1  # +1 for the space
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Ensure we don't have empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    logger.info(f"Split text into {len(chunks)} chunks")
    
    return chunks

def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name to meet ChromaDB requirements:
    1. Contains 3-63 characters
    2. Starts and ends with an alphanumeric character
    3. Contains only alphanumeric characters, underscores or hyphens
    4. Contains no consecutive periods
    5. Is not a valid IPv4 address
    
    Args:
        name: The original collection name
        
    Returns:
        A sanitized valid collection name
    """
    if not name:
        return "default_collection"
        
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Ensure it starts with an alphanumeric character
    if not sanitized[0].isalnum():
        sanitized = "c" + sanitized
        
    # Ensure it ends with an alphanumeric character
    if not sanitized[-1].isalnum():
        sanitized = sanitized + "c"
        
    # Ensure minimum length of 3 characters
    if len(sanitized) < 3:
        sanitized = sanitized + "_collection"
        
    # Ensure maximum length of 63 characters
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
        # Make sure it still ends with alphanumeric
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + "c"
    
    return sanitized

# Initialize the ChromaDB client
def get_chroma_client():
    """
    Initialize and return a ChromaDB client
    """
    try:
        client = chromadb.PersistentClient("./chroma_db")
        return client
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB client: {str(e)}")
        logger.error(f"ChromaDB initialization error: {str(e)}")
        return None

def get_openai_embedding_function():
    """
    Create and return an OpenAI embedding function
    """
    # Use the environment variable for the API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI API key not found in environment variables. Please add it in the Model API Settings.")
        return None
    
    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        return openai_ef
    except Exception as e:
        st.error(f"Failed to initialize OpenAI embedding function: {str(e)}")
        logger.error(f"OpenAI embedding function error: {str(e)}")
        return None

def get_text_with_selenium(url: str) -> str:
    """
    Use Selenium WebDriver to extract text from websites with anti-scraping measures.
    
    Args:
        url: The URL of the website to scrape
        
    Returns:
        Extracted text content as a string
    """
    if not SELENIUM_AVAILABLE:
        raise ImportError("Selenium is not available. Please install it with 'pip install selenium webdriver-manager'")
    
    logger.info(f"Attempting to fetch {url} with Selenium WebDriver")
    
    try:
        # Set up Chrome options with enhanced anti-detection settings
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # More sophisticated user agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0")
        
        # Disable automation flags to prevent detection
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Add referrer to look like we came from Google
        chrome_options.add_argument("--referrer=https://www.google.com/")
        
        chrome_binary = "/nix/store/zi4f80l169xlmivz8vja8wlphq74qqk0-chromium-125.0.6422.141/bin/chromium-browser"
        if os.path.exists(chrome_binary):
            chrome_options.binary_location = chrome_binary
            logger.info(f"Using chromium binary at {chrome_binary}")
        else:
            logger.warning(f"Chromium binary not found at {chrome_binary}")
            # Try alternative locations
            alt_locations = [
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium",
                "/nix/store/zi4f80l169xlmivz8vja8wlphq74qqk0-chromium-125.0.6422.141/bin/chromium"
            ]
            for alt_path in alt_locations:
                if os.path.exists(alt_path):
                    chrome_options.binary_location = alt_path
                    logger.info(f"Using alternative chromium binary at {alt_path}")
                    break
        
        # Try both direct driver initialization and service-based
        driver = None
        try:
            # Try to create the WebDriver directly first (simpler approach)
            driver = webdriver.Chrome(options=chrome_options)
            logger.info("Successfully created Chrome WebDriver directly")
        except Exception as e1:
            logger.warning(f"Direct Chrome initialization failed: {e1}. Trying with ChromeDriverManager.")
            try:
                # Try with ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
                logger.info("Successfully created Chrome WebDriver with ChromeDriverManager")
            except Exception as e2:
                logger.warning(f"ChromeDriverManager initialization failed: {e2}. Trying with default Service.")
                try:
                    # Last resort: try with default Service
                    service = Service()
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    logger.info("Successfully created Chrome WebDriver with default Service")
                except Exception as e3:
                    logger.error(f"All WebDriver initialization methods failed: {e3}")
                    raise Exception("Failed to initialize Chrome WebDriver after multiple attempts")
        
        if not driver:
            raise Exception("Failed to initialize Chrome WebDriver")
            
        try:
            # Set page load timeout
            driver.set_page_load_timeout(60)  # Increased timeout for complex pages
            
            # Add cookies to appear as a returning visitor (domain extracted from URL)
            domain = url.split("//")[1].split("/")[0]  # Extract domain from URL
            try:
                # First navigate to a blank page or the domain root
                driver.get(f"https://{domain}")
                # Add cookie to look like a returning visitor
                driver.add_cookie({
                    "name": "visited_before",
                    "value": "true",
                    "domain": domain
                })
                logger.info(f"Added cookie for domain {domain}")
            except Exception as cookie_error:
                logger.warning(f"Failed to add cookies: {cookie_error}. Continuing anyway.")
            
            # Access the target URL with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    driver.get(url)
                    logger.info(f"Successfully loaded URL on attempt {attempt + 1}")
                    break
                except Exception as load_error:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to load URL on attempt {attempt + 1}: {load_error}. Retrying...")
                        time.sleep(2)  # Wait before retry
                    else:
                        logger.error(f"Failed to load URL after {max_retries} attempts: {load_error}")
                        raise
            
            # Execute JavaScript to scroll down the page to load lazy content
            for scroll in range(5):
                driver.execute_script(f"window.scrollTo(0, {scroll * 500})")
                time.sleep(1)  # Pause between scrolls
            
            # Wait longer for the page to load completely
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except Exception as wait_error:
                logger.warning(f"Wait for body timed out: {wait_error}. Continuing anyway.")
            
            # Try multiple methods to extract content
            text = ""
            
            # Method 1: Direct element text
            try:
                text = driver.find_element(By.TAG_NAME, "body").text
                logger.info("Extracted text using direct body element")
            except Exception as extract_error:
                logger.warning(f"Failed to extract body text: {extract_error}")
            
            # Method 2: If text is empty, try with BeautifulSoup on page source
            if not text.strip():
                logger.info("Direct text extraction yielded no results, trying with BeautifulSoup")
                html_content = driver.page_source
                soup = BeautifulSoup(html_content, "html.parser")
                
                # Remove script and style elements that might contain non-visible text
                for script_or_style in soup(["script", "style", "noscript", "iframe"]):
                    script_or_style.extract()
                
                text = soup.get_text(separator="\n")
                logger.info("Extracted text using BeautifulSoup from page source")
            
            # Process the text (remove extra whitespace)
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
            
            if not text:
                # Method 3: Last resort - try to extract visible texts from specific elements
                logger.info("Previous extraction methods yielded no results, trying with specific elements")
                elements = driver.find_elements(By.XPATH, "//*[not(self::script or self::style or self::noscript)]/text()")
                text = "\n".join([el.text for el in elements if el.text.strip()])
            
            if not text:
                raise ValueError(f"No content extracted from {url} with Selenium after multiple attempts")
                
            logger.info(f"Successfully extracted text from {url} using Selenium")
            return text
        finally:
            # Always close the browser, even if exceptions occur
            if driver:
                try:
                    driver.quit()
                    logger.info("Chrome WebDriver closed successfully")
                except Exception as quit_error:
                    logger.warning(f"Error closing WebDriver: {quit_error}")
    except Exception as e:
        logger.error(f"Error extracting text with Selenium from {url}: {str(e)}")
        raise Exception(f"Failed to extract text with Selenium from {url}: {str(e)}")

def extract_domain(url: str) -> str:
    """
    Extract the base domain from a URL for filtering links in recursive mode
    
    Args:
        url: The URL to extract domain from
        
    Returns:
        The base domain as a string
    """
    try:
        # Add scheme if not present
        if not url.startswith("http"):
            url = "https://" + url
            
        # Parse the URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Remove 'www.' if present
        if domain.startswith("www."):
            domain = domain[4:]
            
        return domain
    except Exception as e:
        logger.error(f"Error extracting domain from {url}: {str(e)}")
        return ""

def extract_links(text_content: str, base_url: str, base_domain: Optional[str] = None) -> List[str]:
    """
    Extract links from text content for recursive crawling
    
    Args:
        text_content: Text content that may contain HTML with links
        base_url: Base URL for resolving relative links
        base_domain: If provided, only return links from this domain
        
    Returns:
        List of absolute URLs found in the content
    """
    try:
        # Try to parse HTML from the text content
        soup = BeautifulSoup(text_content, "html.parser")
        
        # Find all anchor tags with href attributes
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            
            # Skip empty links, anchors, JavaScript links, and mailto links
            if not href or href.startswith(("#", "javascript:", "mailto:")):
                continue
                
            # Convert relative URLs to absolute
            if not href.startswith(("http://", "https://")):
                try:
                    absolute_url = urljoin(base_url, href)
                except Exception:
                    continue
            else:
                absolute_url = href
                
            # Filter by domain if requested
            if base_domain:
                link_domain = extract_domain(absolute_url)
                if not link_domain or link_domain != base_domain:
                    continue
            
            # Add the link if it's not already in the list
            if absolute_url not in links:
                links.append(absolute_url)
        
        return links
    except Exception as e:
        logger.error(f"Error extracting links from content: {str(e)}")
        return []

def get_website_text_content(url: str, use_selenium: bool = False, 
                          use_premium: bool = False, premium_render_js: bool = True,
                          premium_country: str = "") -> str:
    """
    Extract the main text content from a website using various methods
    
    Args:
        url: The URL of the website to scrape
        use_selenium: Whether to use Selenium WebDriver for scraping (helps with anti-scraping measures)
        use_premium: Whether to use premium ScrapingBee service
        premium_render_js: Whether to render JavaScript with ScrapingBee
        premium_country: Country code for ScrapingBee geolocation (e.g., "us")
        
    Returns:
        Extracted text content as a string
        
    Note:
        For advanced multi-page crawling, use the scrape_website function with use_apify=True
        which provides better handling of complex websites through Apify's Website Content Crawler.
    """
    # For websites with strong anti-scraping measures, use special methods first
    if "difc.com" in url or any(domain in url for domain in ["linkedin.com", "instagram.com", "facebook.com"]):
        # Try the advanced proxied scraping method for these highly protected sites
        try:
            text = get_text_with_advanced_proxy(url)
            if text and text.strip():
                logger.info(f"Successfully extracted text from {url} using advanced proxy method")
                return text
        except Exception as proxy_error:
            logger.warning(f"Advanced proxy scraping failed: {str(proxy_error)}. Trying other methods.")
    
    # If selenium is explicitly requested, try that first
    if use_selenium and SELENIUM_AVAILABLE:
        try:
            return get_text_with_selenium(url)
        except Exception as e:
            logger.warning(f"Selenium scraping failed: {str(e)}. Falling back to standard methods.")
            # Fall back to standard methods if selenium fails
    
    # Define headers that mimic a real browser more realistically
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
        "Referer": "https://www.google.com/"  # Add referrer to appear more like a regular browser
    }
    
    try:
        # Try with a rotating proxy approach (retry with different headers)
        try:
            logger.info(f"Attempting to fetch {url} with rotating user agents and headers")
            text = get_text_with_rotating_headers(url)
            if text and text.strip():
                logger.info(f"Successfully extracted text from {url} using rotating headers")
                return text
        except Exception as rotate_error:
            logger.warning(f"Rotating headers method failed: {str(rotate_error)}. Trying standard methods.")
            
        # First try with trafilatura
        logger.info(f"Attempting to fetch {url} with trafilatura")
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                logger.info(f"Successfully extracted text from {url} using trafilatura")
                return text
        
        # If trafilatura fails, try with requests and BeautifulSoup
        logger.info(f"Trafilatura failed, attempting with requests + BeautifulSoup for {url}")
        response = requests.get(url, headers=headers, timeout=15)  # Increased timeout
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch {url}, status code: {response.status_code}")
            
            # Try to fetch the Google cached version as a fallback
            try:
                cached_text = get_google_cached_content(url)
                if cached_text and cached_text.strip():
                    logger.info(f"Successfully extracted text from Google cached version of {url}")
                    return cached_text
            except Exception as cache_error:
                logger.warning(f"Google cache extraction failed: {str(cache_error)}")
            
            # If standard methods fail and selenium is available but wasn't explicitly requested, try it as a last resort
            if SELENIUM_AVAILABLE and not use_selenium:
                logger.info(f"Attempting to fetch {url} with Selenium as a last resort")
                try:
                    return get_text_with_selenium(url)
                except Exception as selenium_e:
                    logger.error(f"Selenium fallback also failed: {str(selenium_e)}")
            
            raise ValueError(f"Failed to download content from {url} (Status code: {response.status_code})")
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator="\n")
        
        # Remove extra whitespace
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        
        if not text:
            raise ValueError(f"No content extracted from {url}")
            
        logger.info(f"Successfully extracted text from {url} using BeautifulSoup")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        raise Exception(f"Failed to extract text from {url}: {str(e)}")

def get_text_with_rotating_headers(url: str) -> str:
    """
    Use rotating sets of headers to try to bypass anti-scraping measures
    
    Args:
        url: The URL of the website to scrape
        
    Returns:
        Extracted text content as a string
    """
    # List of diverse user agents
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 OPR/107.0.0.0",
    ]
    
    # List of referrers to try
    referrers = [
        "https://www.google.com/",
        "https://www.bing.com/",
        "https://www.linkedin.com/",
        "https://twitter.com/",
        "https://www.facebook.com/",
        "https://www.instagram.com/",
    ]
    
    # Combinations of headers to try
    for i in range(5):  # Try up to 5 different combinations
        user_agent = random.choice(user_agents)
        referrer = random.choice(referrers)
        
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": referrer,
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": random.choice(["none", "same-origin", "cross-site"]),
            "Sec-Fetch-User": "?1",
            "Cache-Control": random.choice(["max-age=0", "no-cache"]),
            "Pragma": random.choice(["", "no-cache"]),
            "DNT": random.choice(["1", "0"]),
        }
        
        # Add a cookie that makes it look like we've visited before
        cookies = {"visited_before": "true", "returning_user": "yes"}
        
        try:
            logger.info(f"Trying headers set {i+1} for {url}")
            # Use a slight delay to avoid detection
            time.sleep(random.uniform(1, 3))
            response = requests.get(url, headers=headers, cookies=cookies, timeout=20)
            
            if response.status_code == 200:
                logger.info(f"Successfully fetched {url} with headers set {i+1}")
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style", "noscript", "iframe"]):
                    script.extract()
                
                # Get text content
                text = soup.get_text(separator="\n")
                text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
                
                if text:
                    return text
                else:
                    logger.warning(f"No text content extracted from {url} with headers set {i+1}")
        except Exception as e:
            logger.warning(f"Error with headers set {i+1} for {url}: {str(e)}")
    
    raise Exception(f"All header rotation attempts failed for {url}")

def get_google_cached_content(url: str) -> str:
    """
    Attempt to get content from Google's cached version of the page
    
    Args:
        url: The URL of the website to scrape
        
    Returns:
        Extracted text content as a string
    """
    # Encode the URL for use in Google cache URL
    encoded_url = urllib.parse.quote(url, safe='')
    cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{encoded_url}"
    
    logger.info(f"Attempting to fetch Google cached version: {cache_url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }
    
    try:
        response = requests.get(cache_url, headers=headers, timeout=20)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch Google cache, status code: {response.status_code}")
            return ""
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Look for the cached content (it's typically in a specific frame/div)
        cache_content = soup.select_one("div.page-content")
        if not cache_content:
            cache_content = soup  # Fallback to entire document
        
        # Remove irrelevant elements
        for element in cache_content(["script", "style", "noscript", "iframe"]):
            element.extract()
        
        # Get text content
        text = cache_content.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        
        return text
    except Exception as e:
        logger.warning(f"Error fetching Google cache for {url}: {str(e)}")
        return ""

def get_text_with_advanced_proxy(url: str) -> str:
    """
    Use ScrapingBee API to extract content from sites with strong anti-scraping measures.
    This method leverages ScrapingBee's premium capabilities to handle sites with:
    - JavaScript rendering
    - Anti-bot protection
    - CAPTCHAs
    - IP blocking
    
    Args:
        url: The URL of the website to scrape
        
    Returns:
        Extracted text content as a string
    """
    # Check if ScrapingBee API key is available
    scrapingbee_api_key = os.environ.get("SCRAPINGBEE_API_KEY")
    if not scrapingbee_api_key:
        raise Exception("ScrapingBee API key not found. Please add it to environment variables or through the UI.")
    
    logger.info(f"Using ScrapingBee API to scrape {url}")
    
    try:
        # First, check the cache
        safe_url = re.sub(r'[^\w]', '_', url)
        cache_file_name = f"cached_{safe_url}.txt"
        cache_path = os.path.join(".", cache_file_name)
        
        # Check if we have a recent cached version (less than 24 hours old)
        if os.path.exists(cache_path):
            # Check file age
            file_age = time.time() - os.path.getmtime(cache_path)
            if file_age < 86400:  # 24 hours in seconds
                with open(cache_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        logger.info(f"Using cached content for {url} (cached {int(file_age/3600)} hours ago)")
                        return content
        
        # Build the ScrapingBee API URL with parameters
        api_url = "https://app.scrapingbee.com/api/v1/"
        params = {
            "api_key": scrapingbee_api_key,
            "url": url,
            "render_js": "true",           # Enable JavaScript rendering
            "premium_proxy": "true",       # Use premium proxies
            "country_code": "",            # Can be set to specific country if needed
            "wait": "2000",                # Wait for page to fully load (ms)
            "block_resources": "false",    # Allow all resources to load
            "return_page_source": "true",  # Return full HTML
            "extract_rules": '{"text_content": {"selector": "body", "type": "text"}}' # Extract text content
        }
        
        # Make the API request
        response = requests.get(api_url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            # If the extract_rules parameter worked, we'll have JSON with text_content
            try:
                json_response = response.json()
                if json_response and "text_content" in json_response:
                    extracted_text = json_response["text_content"]
                    logger.info(f"Successfully extracted {len(extracted_text)} characters from {url} using ScrapingBee")
                else:
                    # If JSON extraction failed, parse HTML with trafilatura
                    html_content = response.text
                    extracted_text = trafilatura.extract(html_content)
                    logger.info(f"Extracted {len(extracted_text)} characters from HTML using trafilatura")
            except:
                # If it's not JSON, try parsing the HTML with trafilatura
                html_content = response.text
                extracted_text = trafilatura.extract(html_content)
                logger.info(f"Extracted {len(extracted_text)} characters from HTML using trafilatura")
            
            # Cache the extracted text
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
                
            return extracted_text
        else:
            logger.error(f"ScrapingBee API error: HTTP {response.status_code} - {response.text}")
            raise Exception(f"ScrapingBee API error: HTTP {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error in ScrapingBee API method for {url}: {str(e)}")
        raise Exception(f"Failed to extract content using ScrapingBee: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of specified size
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
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

def create_vector_collection(collection_name: str) -> Union[chromadb.Collection, None]:
    """
    Create or get a ChromaDB collection for storing vectors
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        ChromaDB collection or None if failed
    """
    client = get_chroma_client()
    if not client:
        return None
    
    embedding_function = get_openai_embedding_function()
    if not embedding_function:
        return None
    
    # Sanitize the collection name to ensure it meets ChromaDB requirements
    sanitized_name = sanitize_collection_name(collection_name)
    
    try:
        # Try to get existing collection, or create a new one
        try:
            collection = client.get_collection(name=sanitized_name, embedding_function=embedding_function)
            logger.info(f"Retrieved existing collection: {sanitized_name} (original: {collection_name})")
        except:
            collection = client.create_collection(name=sanitized_name, embedding_function=embedding_function)
            logger.info(f"Created new collection: {sanitized_name} (original: {collection_name})")
        
        return collection
    except Exception as e:
        st.error(f"Failed to create/get vector collection: {str(e)}")
        logger.error(f"Vector collection error: {str(e)}")
        return None

def add_text_to_collection(
    collection: chromadb.Collection, 
    chunks: List[str], 
    source_url: str,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Add text chunks to a ChromaDB collection
    
    Args:
        collection: ChromaDB collection
        chunks: List of text chunks to add
        source_url: Source URL for the chunks
        metadata: Additional metadata to store with each chunk
        
    Returns:
        Boolean indicating success
    """
    if not chunks:
        logger.warning("No chunks to add to vector database")
        return False
    
    try:
        # Create document IDs, texts and metadatas
        ids = [f"{source_url}_{i}" for i in range(len(chunks))]
        
        # Create metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "source": source_url,
                "chunk_id": i,
                "chunk_count": len(chunks),
                "timestamp": time.time(),
            }
            
            # Add additional metadata if provided
            if metadata:
                # Sanitize metadata by removing None values (ChromaDB doesn't accept None values)
                sanitized_metadata = {k: v for k, v in metadata.items() if v is not None}
                chunk_metadata.update(sanitized_metadata)
                
            metadatas.append(chunk_metadata)
        
        # Add documents to collection
        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector collection")
        return True
    except Exception as e:
        st.error(f"Failed to add chunks to vector collection: {str(e)}")
        logger.error(f"Vector insertion error: {str(e)}")
        return False

def query_collection(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5
) -> Tuple[List[str], List[Dict]]:
    """
    Query the vector collection for similar documents
    
    Args:
        collection: ChromaDB collection
        query: Query string
        n_results: Number of results to return
        
    Returns:
        Tuple of (documents, metadatas)
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        return documents, metadatas
    except Exception as e:
        st.error(f"Failed to query vector collection: {str(e)}")
        logger.error(f"Vector query error: {str(e)}")
        return [], []

def get_all_collections():
    """
    Get a list of all available collections
    
    Returns:
        List of collection names
    """
    client = get_chroma_client()
    if not client:
        return []
    
    try:
        # Handle different versions of ChromaDB
        try:
            # In newer versions of ChromaDB, list_collections returns just the names
            collection_names = client.list_collections()
            if collection_names and isinstance(collection_names[0], str):
                logger.info("Using ChromaDB v0.6+ collection names")
                return collection_names
        except Exception as inner_e:
            logger.warning(f"Error with newer ChromaDB API: {str(inner_e)}")
        
        # Fall back to older version approach
        try:
            collections = client.list_collections()
            return [c.name for c in collections]
        except Exception as inner_e2:
            logger.warning(f"Error with older ChromaDB API: {str(inner_e2)}")
            
        # Last resort, attempt to get known collections
        known_collections = ["difc_website", "default_collection"]
        found_collections = []
        for name in known_collections:
            try:
                client.get_collection(name)
                found_collections.append(name)
            except:
                pass
                
        if found_collections:
            logger.info(f"Retrieved collections through direct access: {found_collections}")
            return found_collections
            
        return []
    except Exception as e:
        logger.error(f"Error getting collections: {str(e)}")
        return []

def delete_collection(collection_name: str) -> bool:
    """
    Delete a collection
    
    Args:
        collection_name: Name of the collection to delete
        
    Returns:
        Boolean indicating success
    """
    client = get_chroma_client()
    if not client:
        return False
    
    # Sanitize the collection name
    sanitized_name = sanitize_collection_name(collection_name)
    
    try:
        client.delete_collection(sanitized_name)
        logger.info(f"Deleted collection: {sanitized_name} (original: {collection_name})")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        return False

def get_collection_stats(collection_name: str) -> Dict:
    """
    Get statistics about a collection
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        Dictionary with collection statistics
    """
    client = get_chroma_client()
    if not client:
        return {}
    
    # Sanitize the collection name
    sanitized_name = sanitize_collection_name(collection_name)
    
    try:
        collection = client.get_collection(sanitized_name)
        count = collection.count()
        
        # Get a sample of the collection to analyze
        if count > 0:
            sample = min(count, 10)
            sample_docs = collection.get(limit=sample)
            
            # Extract source URLs
            sources = set()
            for metadata in sample_docs.get("metadatas", []):
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
            
            return {
                "name": collection_name,  # Return original name for display
                "sanitized_name": sanitized_name,  # Include sanitized name for reference
                "document_count": count,
                "sources": list(sources)
            }
        
        return {
            "name": collection_name,
            "sanitized_name": sanitized_name,
            "document_count": count,
            "sources": []
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        return {}

def scrape_and_store_website(url: str, collection_name: str, metadata: Optional[Dict] = None, use_selenium: bool = False) -> bool:
    """
    Scrape a website, process the text, and store in vector database
    
    Args:
        url: Website URL to scrape
        collection_name: Name of the collection to store in
        metadata: Additional metadata to store
        use_selenium: Whether to use Selenium WebDriver for scraping (helps with anti-scraping measures)
        
    Returns:
        Boolean indicating success
    """
    try:
        # Get website content
        text_content = get_website_text_content(url, use_selenium=use_selenium)
        
        # Create chunks from text
        chunks = chunk_text(text_content)
        
        if not chunks:
            st.error(f"No content extracted from {url}")
            logger.error(f"No content extracted from {url}")
            return False
            
        # Create or get collection
        collection = create_vector_collection(collection_name)
        if not collection:
            return False
        
        # Add chunks to collection
        return add_text_to_collection(collection, chunks, url, metadata)
    except Exception as e:
        st.error(f"Failed to scrape and store website: {str(e)}")
        logger.error(f"Scraping error: {str(e)}")
        return False

def search_website_content(query: str, collection_name: str, n_results: int = 5) -> pd.DataFrame:
    """
    Search for website content related to the query
    
    Args:
        query: Search query
        collection_name: Name of the collection to search
        n_results: Number of results to return
        
    Returns:
        DataFrame with search results
    """
    # Create or get collection
    collection = create_vector_collection(collection_name)
    if not collection:
        return pd.DataFrame()
    
    # Query collection
    documents, metadatas = query_collection(collection, query, n_results)
    
    # Create DataFrame from results
    results = []
    for doc, meta in zip(documents, metadatas):
        results.append({
            "content": doc,
            "source": meta.get("source", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "timestamp": meta.get("timestamp", "")
        })
    
    return pd.DataFrame(results)

def generate_query_variations(query: str, model_provider: str = "openai", num_variations: int = 3) -> List[str]:
    """
    Generate multiple variations of a query to improve retrieval
    
    Args:
        query: Original user query
        model_provider: AI provider to use
        num_variations: Number of query variations to generate
        
    Returns:
        List of query variations
    """
    from model_clients import get_model_client
    
    # Get AI model client
    client = get_model_client(model_provider)
    if not client:
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
        
        response = client.generate_completion(messages, temperature=0.7)
        
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

def search_website_with_multiquery(query: str, collection_name: str, n_results: int = 5, model_provider: str = "openai") -> pd.DataFrame:
    """
    Search for website content using multiple query variations to improve retrieval
    
    Args:
        query: Original user query
        collection_name: Name of the collection to search
        n_results: Number of results to return per query variation
        model_provider: AI provider to use for generating query variations
        
    Returns:
        DataFrame with combined search results
    """
    # Generate query variations
    query_variations = generate_query_variations(query, model_provider)
    
    # Search using each query variation
    all_results = []
    for variation in query_variations:
        df = search_website_content(variation, collection_name, n_results)
        if not df.empty:
            all_results.append(df)
    
    if not all_results:
        return pd.DataFrame()  # Return empty DataFrame if no results
    
    # Combine results and remove duplicates
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["content"])
    
    # Sort by relevance (assuming most relevant results appear in more query results)
    # This is a simple heuristic and could be improved with actual relevance scores
    return combined_df.head(n_results * 2)  # Return more results than single query

def ask_question_about_website(query: str, collection_name: Union[str, List[str]], model_provider: str = "openai", use_streaming: bool = True) -> str:
    """
    Ask a question about a website using AI and vector search with improved retrieval
    
    Args:
        query: User's question
        collection_name: Name of the collection to search or list of collection names
        model_provider: AI provider to use (openai, anthropic, mistral)
        use_streaming: Whether to use streaming response (if supported by provider)
        
    Returns:
        AI-generated answer or a streaming placeholder if streaming is enabled
    """
    # Import here to avoid circular imports
    from model_clients import get_model_client
    import streamlit as st
    
    # Use enhanced retrieval if available
    try:
        from enhanced_retrieval import WebContentRetrieval
        enhanced_retriever = WebContentRetrieval(model_provider=model_provider)
        
        # Use the comprehensive enhanced search
        combined_df = enhanced_retriever.enhanced_web_search(query, collection_name)
        
        if combined_df.empty:
            return "I couldn't find any relevant information to answer your question."
            
        # Limit to top results to avoid context window issues
        combined_df = combined_df.head(10)
        
        # Get the collections used
        collections = combined_df['collection'].unique().tolist()
            
    except (ImportError, Exception) as e:
        logger.warning(f"Enhanced retrieval not available, falling back to standard retrieval: {str(e)}")
        
        # Fall back to standard multi-query retrieval
        # Handle multiple collections
        all_results = []
        
        # Convert single collection to list for consistent processing
        collections = collection_name if isinstance(collection_name, list) else [collection_name]
        
        # Search each collection and combine results
        for collection in collections:
            df = search_website_with_multiquery(query, collection, n_results=5, model_provider=model_provider)
            if not df.empty:
                # Add collection name to the results for reference
                df['collection'] = collection
                all_results.append(df)
        
        if not all_results:
            return "I couldn't find any relevant information to answer your question."
            
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=["content"])
        
        # Sort by relevance score if available, otherwise keep original order
        if 'relevance_score' in combined_df.columns:
            combined_df = combined_df.sort_values(by='relevance_score', ascending=False)
        
        # Limit to top results to avoid context window issues
        combined_df = combined_df.head(10)
    
    # Concatenate all retrieved content
    # If we have position information from hierarchical retrieval, arrange properly
    if 'position' in combined_df.columns:
        # Group by source and arrange by position
        context_parts = []
        for source in combined_df['source'].unique():
            source_df = combined_df[combined_df['source'] == source]
            
            # Check if we have a hierarchical set
            if set(['previous', 'main', 'next']).intersection(set(source_df['position'])):
                # Sort by position (previous, main, next)
                position_order = {'previous': 0, 'main': 1, 'next': 2}
                source_df['pos_order'] = source_df['position'].map(lambda x: position_order.get(x, 3))
                source_df = source_df.sort_values(['chunk_id', 'pos_order'])
                
                # Format with the source
                content = f"=== From {source} ===\n" + "\n\n".join(source_df['content'].tolist())
                context_parts.append(content)
            else:
                # Regular content
                content = f"=== From {source} ===\n" + "\n\n".join(source_df['content'].tolist())
                context_parts.append(content)
                
        context = "\n\n".join(context_parts)
    else:
        # Standard concatenation
        context = "\n\n".join(combined_df["content"].tolist())
    
    # Create an improved prompt for the AI with better instructions
    collection_info = ""
    if len(collections) > 1:
        collection_info = f"The context contains information extracted from {len(collections)} different websites: {', '.join(collections)}. "
    else:
        collection_info = f"The context contains information extracted from the website: {collections[0]}. "
        
    prompt = f"""
    You are tasked with answering a question based on the context provided below. {collection_info}
    
    Question: {query}
    
    Context:
    {context}
    
    Instructions:
    1. Answer the question based ONLY on the information in the context
    2. If the context doesn't contain enough information to provide a complete answer, explain what information is available and what's missing
    3. If the context contains contradictory information, point this out
    4. Use bullet points if it helps organize the answer
    5. Cite the specific parts of the context that support your answer
    6. Be concise and clear
    
    Answer:
    """
    
    # Get AI model client
    client = get_model_client(model_provider)
    if not client:
        return "Unable to connect to AI model. Please check your API key settings."
    
    # Create messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in extracting and synthesizing information from web content. Your answers should be accurate, comprehensive, and based solely on the provided context."},
        {"role": "user", "content": prompt}
    ]
    
    if use_streaming and model_provider == "openai":  # Currently only OpenAI supports streaming
        try:
            # Create a placeholder for streaming output
            answer_placeholder = st.empty()
            answer_placeholder.write("Generating response...")
            
            # Start with empty response
            full_response = ""
            
            # Get streaming response
            for text_chunk in client.generate_streaming_completion(messages, temperature=0.5):
                # Append to the full response
                full_response += text_chunk
                
                # Update the placeholder with the current cumulative response
                answer_placeholder.markdown(full_response)
            
            # The placeholder will have been updated with the full response
            return full_response
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            # Fall back to non-streaming response
            try:
                response = client.generate_completion(messages, temperature=0.5)
                return response
            except Exception as e2:
                logger.error(f"Error in fallback response: {str(e2)}")
                return f"Error generating response: {str(e2)}"
    else:
        # Non-streaming response (default or fallback)
        try:
            # Use slightly higher temperature for more comprehensive responses
            response = client.generate_completion(messages, temperature=0.5)
            return response
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return f"Error generating response: {str(e)}"