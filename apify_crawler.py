"""
Apify Website Content Crawler Module

This module integrates with the Apify platform to provide advanced website crawling
functionality. It uses the Website Content Crawler actor to extract content from
entire websites in an LLM-ready format.
"""

import os
import time
import logging
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from apify_client import ApifyClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Apify client with API token from environment
APIFY_API_KEY = os.getenv("APIFY_API_KEY")
apify_client = ApifyClient(APIFY_API_KEY) if APIFY_API_KEY else None

# Website Content Crawler actor ID
WEBSITE_CONTENT_CRAWLER_ACTOR_ID = "apify/website-content-crawler"


class ApifyCrawler:
    """
    A class that handles crawling websites using Apify's Website Content Crawler.
    """

    def __init__(self):
        """Initialize the Apify crawler with API client."""
        self.client = apify_client
        if not self.client:
            logger.warning("Apify API key not found. Set APIFY_API_KEY environment variable.")

    def is_available(self) -> bool:
        """Check if Apify crawler is available (API key is set)."""
        return self.client is not None

    def crawl_website(self, url: str, max_crawl_pages: int = 20, 
                      max_crawl_depth: int = 5, exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Crawl a website using Apify's Website Content Crawler actor.
        
        Args:
            url: The starting URL to crawl
            max_crawl_pages: Maximum number of pages to crawl
            max_crawl_depth: Maximum crawl depth
            exclude_patterns: List of URL patterns to exclude from crawling
            
        Returns:
            Dictionary with crawl results and status
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Apify API key not set. Cannot crawl website.",
                "data": None
            }
            
        # Validate URL
        if not url.startswith("http"):
            url = f"https://{url}"
            
        # Prepare input for the actor
        run_input = {
            "startUrls": [{"url": url}],
            "maxCrawlPages": max_crawl_pages,
            "maxCrawlDepth": max_crawl_depth,
            "saveHtml": False,  # We don't need HTML for LLM processing
            "saveMarkdown": True,  # Markdown is good for LLM
            "saveScreenshots": False,  # We don't need screenshots
            "includeUrlGlobs": [f"{url}/**"],  # Stay on the same domain by default
        }
        
        # Add exclusion patterns if provided
        if exclude_patterns:
            run_input["excludeUrlGlobs"] = exclude_patterns
            
        try:
            logger.info(f"Starting Apify crawler for URL: {url}")
            
            # Start the actor and wait for it to finish
            run = self.client.actor(WEBSITE_CONTENT_CRAWLER_ACTOR_ID).call(run_input=run_input)
            
            # Get the dataset ID from the run
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                return {
                    "success": False,
                    "error": "Failed to get dataset ID from Apify run",
                    "data": None
                }
                
            # Get results from the dataset
            items = self.client.dataset(dataset_id).list_items().items
            
            # Extract relevant data
            crawl_results = {
                "pages": [],
                "url": url,
                "crawl_time": time.time(),
                "page_count": len(items)
            }
            
            # Process each crawled page
            for item in items:
                crawl_results["pages"].append({
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "text": item.get("markdown") or item.get("text"),
                    "headings": item.get("headings", []),
                    "links": item.get("links", [])
                })
                
            logger.info(f"Completed crawling {len(crawl_results['pages'])} pages from {url}")
            
            return {
                "success": True,
                "data": crawl_results
            }
            
        except Exception as e:
            logger.error(f"Error crawling website with Apify: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    def get_active_runs(self) -> List[Dict[str, Any]]:
        """Get a list of currently active runs on the Apify platform."""
        if not self.is_available():
            return []
            
        try:
            # Get all runs for the Website Content Crawler actor
            runs = self.client.actor(WEBSITE_CONTENT_CRAWLER_ACTOR_ID).runs().list(desc=True).items
            
            # Filter to only running/ready states
            active_runs = [
                run for run in runs 
                if run.get("status") in ("RUNNING", "READY", "CREATED")
            ]
            
            return active_runs
        except Exception as e:
            logger.error(f"Error getting active Apify runs: {str(e)}")
            return []
    
    def stop_run(self, run_id: str) -> bool:
        """Stop a specific run by its ID."""
        if not self.is_available():
            return False
            
        try:
            # Abort the run
            self.client.run(run_id).abort()
            return True
        except Exception as e:
            logger.error(f"Error stopping Apify run {run_id}: {str(e)}")
            return False
    
    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get the status of a specific run."""
        if not self.is_available():
            return {"status": "ERROR", "error": "Apify API key not set"}
            
        try:
            run = self.client.run(run_id).get()
            return run
        except Exception as e:
            logger.error(f"Error getting status for run {run_id}: {str(e)}")
            return {"status": "ERROR", "error": str(e)}


# Create a singleton instance
apify_crawler = ApifyCrawler()


def get_crawl_data_as_documents(crawl_results: Dict[str, Any], chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Convert Apify crawl results into document format suitable for vector storage.
    
    Args:
        crawl_results: Results from the Apify crawler
        chunk_size: Maximum size for each document chunk
        
    Returns:
        List of document dictionaries with content and metadata
    """
    if not crawl_results or not crawl_results.get("success"):
        return []
        
    data = crawl_results.get("data", {})
    pages = data.get("pages", [])
    
    documents = []
    
    for page in pages:
        page_url = page.get("url", "")
        page_title = page.get("title", "")
        page_text = page.get("text", "")
        
        if not page_text:
            continue
            
        # Split text into chunks if it's too large
        if len(page_text) > chunk_size:
            chunks = [page_text[i:i+chunk_size] for i in range(0, len(page_text), chunk_size)]
        else:
            chunks = [page_text]
            
        # Create a document for each chunk
        for i, chunk in enumerate(chunks):
            # Sanitize metadata to ensure no None values
            metadata = {
                "url": page_url or "",
                "title": page_title or "",
                "chunk": i,
                "source": "apify",
                "total_chunks": len(chunks)
            }
            
            # Replace any None values with empty strings to prevent ChromaDB errors
            metadata = {k: v if v is not None else "" for k, v in metadata.items()}
            
            documents.append({
                "content": chunk,
                "metadata": metadata
            })
    
    return documents


def save_crawl_results(crawl_results: Dict[str, Any], output_dir: str = "./crawl_data") -> str:
    """
    Save crawl results to disk for later retrieval.
    
    Args:
        crawl_results: Results from the Apify crawler
        output_dir: Directory to save results
        
    Returns:
        Path to the saved results file
    """
    if not crawl_results or not crawl_results.get("success"):
        return ""
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    data = crawl_results.get("data", {})
    url = data.get("url", "unknown")
    
    # Create a filename based on the URL and timestamp
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
    timestamp = int(time.time())
    filename = f"apify_crawl_{url_hash}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save the results to disk
    with open(filepath, "w") as f:
        json.dump(crawl_results, f, indent=2)
        
    return filepath


if __name__ == "__main__":
    # Example usage
    test_url = "https://www.example.com"
    crawler = ApifyCrawler()
    
    if crawler.is_available():
        results = crawler.crawl_website(test_url, max_crawl_pages=5)
        if results["success"]:
            print(f"Successfully crawled {len(results['data']['pages'])} pages")
            filepath = save_crawl_results(results)
            print(f"Results saved to {filepath}")
        else:
            print(f"Crawl failed: {results.get('error')}")
    else:
        print("Apify API key not set. Please set the APIFY_API_KEY environment variable.")