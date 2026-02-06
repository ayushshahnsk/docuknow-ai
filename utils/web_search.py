"""
Web Search for DocuKnow AI (Updated for Version 1 - Multilingual)

Purpose:
- Unified web search interface with multilingual support
- Real-time information from multiple sources
- Free APIs with fallback mechanisms
- Backward compatibility with existing Tavily API
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import time

# Try to import our new multilingual search
try:
    from utils.multilingual_search import web_search as multilingual_search, get_search_engine
    MULTILINGUAL_SEARCH_AVAILABLE = True
except ImportError as e:
    MULTILINGUAL_SEARCH_AVAILABLE = False
    print(f"Multilingual search not available: {e}")
    print("Install required packages: pip install googletrans==3.1.0a0 requests")

# Try to import Tavily for backward compatibility
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("Tavily not available. Install: pip install tavily-python")

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration from environment or defaults
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "10"))
MAX_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "3"))
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "hybrid").lower()

# Rate limiting to avoid API abuse
class RateLimiter:
    def __init__(self, calls_per_minute=30):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        if len(self.call_times) >= self.calls_per_minute:
            # Calculate wait time
            oldest_call = self.call_times[0]
            wait_time = 60 - (now - oldest_call)
            if wait_time > 0:
                time.sleep(wait_time)
        
        self.call_times.append(now)

# Global rate limiter
_rate_limiter = RateLimiter(calls_per_minute=20)

def web_search(
    query: str, 
    api_key: str | None = None, 
    lang: str = "en",
    provider: Optional[str] = None,
    max_results: int = 3
) -> Optional[str]:
    """
    Enhanced web search with multilingual support.
    
    Search priority:
    1. Our multilingual search (free, real-time)
    2. Tavily API (if key provided)
    3. Direct DuckDuckGo fallback
    
    Args:
        query: Search query
        api_key: Optional API key (for Tavily or other paid services)
        lang: Language code for search results (e.g., 'en', 'hi', 'fr')
        provider: Force specific provider ('multilingual', 'tavily', 'duckduckgo')
        max_results: Maximum number of results to return
        
    Returns:
        Search results as string or None
    """
    if not query or len(query.strip()) < 2:
        logger.warning("Query too short for web search")
        return None
    
    # Apply rate limiting
    _rate_limiter.wait_if_needed()
    
    logger.info(f"Web search: '{query}' (lang: {lang}, provider: {provider or SEARCH_PROVIDER})")
    
    # Normalize language code
    lang = lang.lower()
    if lang not in ['en', 'hi', 'mr', 'gu', 'fr', 'bn', 'de', 'zh', 'es', 'ta', 'te', 'kn', 'ml', 'pa', 'ur', 'ar', 'ja', 'ko', 'ru', 'pt', 'it']:
        logger.warning(f"Unsupported language: {lang}, defaulting to English")
        lang = 'en'
    
    # Determine provider to use
    actual_provider = provider or SEARCH_PROVIDER
    
    # Provider 1: Multilingual Search (Recommended - Free)
    if actual_provider in ["multilingual", "hybrid"] and MULTILINGUAL_SEARCH_AVAILABLE:
        try:
            logger.debug(f"Using multilingual search for: {query}")
            result = multilingual_search(query, api_key, lang)
            
            if result:
                logger.info(f"Multilingual search successful for '{query}'")
                
                # Format result with source info
                formatted_result = f"{result}\n\n[Source: Web Search | Language: {lang.upper()}]"
                return formatted_result[:2000]  # Limit length
                
        except Exception as e:
            logger.error(f"Multilingual search failed: {e}")
            # Fall through to next provider
    
    # Provider 2: Tavily API (if API key is provided)
    if actual_provider in ["tavily", "hybrid"] and TAVILY_AVAILABLE and api_key:
        try:
            logger.debug(f"Using Tavily API for: {query}")
            client = TavilyClient(api_key=api_key)
            
            # Tavily search parameters
            search_params = {
                "query": query,
                "search_depth": "basic",
                "max_results": min(max_results, 5),
                "include_answer": True,
                "include_raw_content": False,
            }
            
            # Try to set language if supported
            if lang == "en":  # Tavily primarily supports English
                response = client.search(**search_params)
                
                if response and "results" in response:
                    results = response["results"]
                    
                    # Use answer if available
                    if response.get("answer"):
                        result_text = response["answer"]
                    else:
                        # Combine top results
                        answers = []
                        for r in results[:max_results]:
                            if "content" in r:
                                answers.append(r["content"])
                            elif "snippet" in r:
                                answers.append(r["snippet"])
                        
                        if answers:
                            result_text = "\n\n".join(answers)
                        else:
                            result_text = None
                    
                    if result_text:
                        logger.info(f"Tavily search successful for '{query}'")
                        formatted_result = f"{result_text}\n\n[Source: Tavily API | Language: {lang.upper()}]"
                        return formatted_result[:2000]
                        
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            # Fall through to next provider
    
    # Provider 3: Direct DuckDuckGo (Fallback - Always available)
    if actual_provider in ["duckduckgo", "hybrid", "fallback"]:
        try:
            logger.debug(f"Using DuckDuckGo fallback for: {query}")
            import requests
            
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
                "t": "docuknow_ai",
                "kl": "wt-wt"  # Region/language (wt-wt = no region)
            }
            
            response = requests.get(url, params=params, timeout=SEARCH_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                result_parts = []
                
                # Check for Abstract
                if data.get("AbstractText"):
                    result_parts.append(data["AbstractText"])
                
                # Check for Definition
                elif data.get("Definition"):
                    result_parts.append(data["Definition"])
                
                # Check for Related Topics
                elif data.get("RelatedTopics"):
                    topics = []
                    for topic in data["RelatedTopics"][:max_results]:
                        if isinstance(topic, dict) and "Text" in topic:
                            topics.append(topic["Text"])
                        elif isinstance(topic, str):
                            topics.append(topic)
                    
                    if topics:
                        result_parts.append("\n".join(topics))
                
                # Check for Answer
                elif data.get("Answer"):
                    result_parts.append(data["Answer"])
                
                if result_parts:
                    result_text = "\n\n".join(result_parts)
                    logger.info(f"DuckDuckGo fallback successful for '{query}'")
                    
                    # Simple translation hint for non-English queries
                    translation_note = ""
                    if lang != "en":
                        translation_note = f"\n\nNote: Results are in English. Original query was in {lang.upper()}."
                    
                    formatted_result = f"{result_text}{translation_note}\n\n[Source: DuckDuckGo | Language: EN]"
                    return formatted_result[:2000]
                    
        except Exception as e:
            logger.error(f"DuckDuckGo fallback failed: {e}")
    
    # Provider 4: Wikipedia Direct (For factual queries)
    if actual_provider in ["wikipedia", "hybrid"] and "what" in query.lower() or "who" in query.lower():
        try:
            logger.debug(f"Trying Wikipedia direct for: {query}")
            import requests
            
            # Try English Wikipedia first
            wiki_lang = "en" if lang not in ["es", "fr", "de", "it", "pt", "ru", "zh", "ja", "hi"] else lang
            
            # Search Wikipedia
            search_url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=SEARCH_TIMEOUT)
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                
                if search_data["query"]["search"]:
                    page_id = search_data["query"]["search"][0]["pageid"]
                    
                    # Get page extract
                    extract_url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
                    extract_params = {
                        "action": "query",
                        "pageids": page_id,
                        "prop": "extracts",
                        "exintro": "1",
                        "explaintext": "1",
                        "format": "json"
                    }
                    
                    extract_response = requests.get(extract_url, params=extract_params, timeout=SEARCH_TIMEOUT)
                    
                    if extract_response.status_code == 200:
                        extract_data = extract_response.json()
                        pages = extract_data["query"]["pages"]
                        
                        if str(page_id) in pages and "extract" in pages[str(page_id)]:
                            result_text = pages[str(page_id)]["extract"][:1500]
                            logger.info(f"Wikipedia direct successful for '{query}'")
                            
                            formatted_result = f"{result_text}\n\n[Source: Wikipedia | Language: {wiki_lang.upper()}]"
                            return formatted_result
                            
        except Exception as e:
            logger.error(f"Wikipedia direct failed: {e}")
    
    # All providers failed
    logger.warning(f"No web search results found for: '{query}'")
    return None


def advanced_web_search(
    query: str,
    lang: str = "en",
    search_depth: str = "basic",  # "basic" or "advanced"
    include_images: bool = False,
    include_news: bool = False,
    timeframe: Optional[str] = None,  # "day", "week", "month", "year"
) -> Dict[str, Any]:
    """
    Advanced web search with more options.
    
    Args:
        query: Search query
        lang: Language code
        search_depth: Search depth level
        include_images: Whether to include image URLs
        include_news: Whether to include news results
        timeframe: Timeframe for results
        
    Returns:
        Dictionary with structured search results
    """
    # Default result structure
    result = {
        "query": query,
        "language": lang,
        "success": False,
        "results": [],
        "sources": [],
        "timestamp": time.time(),
        "error": None
    }
    
    try:
        # Use our multilingual search engine for advanced search
        if MULTILINGUAL_SEARCH_AVAILABLE:
            searcher = get_search_engine()
            
            # Perform search
            search_result = searcher.search(query, lang, use_cache=False)
            
            if search_result:
                result["success"] = True
                result["results"] = [{
                    "content": search_result,
                    "type": "text",
                    "source": "multilingual_search"
                }]
                result["sources"] = ["multilingual_search"]
                
                # Try to get additional info if requested
                if include_news or "news" in query.lower():
                    news_result = searcher.news_search(query, lang)
                    if news_result:
                        result["results"].append({
                            "content": news_result,
                            "type": "news",
                            "source": "news_search"
                        })
                        result["sources"].append("news_search")
        
        # Fallback to simple search
        if not result["success"]:
            simple_result = web_search(query, lang=lang)
            if simple_result:
                result["success"] = True
                result["results"] = [{
                    "content": simple_result,
                    "type": "text",
                    "source": "web_search"
                }]
                result["sources"] = ["web_search"]
        
        return result
        
    except Exception as e:
        logger.error(f"Advanced web search failed: {e}")
        result["error"] = str(e)
        return result


def clear_search_cache():
    """Clear all search caches."""
    try:
        if MULTILINGUAL_SEARCH_AVAILABLE:
            searcher = get_search_engine()
            searcher.clear_cache()
            logger.info("Search cache cleared")
    except Exception as e:
        logger.error(f"Failed to clear search cache: {e}")


def get_search_stats() -> Dict[str, Any]:
    """Get search engine statistics."""
    stats = {
        "multilingual_search_available": MULTILINGUAL_SEARCH_AVAILABLE,
        "tavily_available": TAVILY_AVAILABLE,
        "default_provider": SEARCH_PROVIDER,
        "rate_limit": "20 calls/minute",
    }
    
    try:
        if MULTILINGUAL_SEARCH_AVAILABLE:
            searcher = get_search_engine()
            cache_stats = searcher.get_cache_stats()
            stats["cache_stats"] = cache_stats
    except:
        pass
    
    return stats


# Test function
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("Testing Web Search Module")
    print("=" * 60)
    
    # Test queries in different languages
    test_queries = [
        ("What is artificial intelligence?", "en"),
        ("कृत्रिम बुद्धिमत्ता क्या है?", "hi"),
        ("Qu'est-ce que l'intelligence artificielle?", "fr"),
        ("今天天气怎么样？", "zh"),
    ]
    
    for query, lang in test_queries:
        print(f"\nQuery: {query}")
        print(f"Language: {lang}")
        print("-" * 40)
        
        result = web_search(query, lang=lang)
        
        if result:
            print(f"Success! Result length: {len(result)} characters")
            print(f"Preview: {result[:200]}...")
        else:
            print("No results found")
        
        time.sleep(2)  # Rate limiting
    
    # Test advanced search
    print("\n\nTesting Advanced Search")
    print("=" * 60)
    
    advanced_result = advanced_web_search(
        query="latest AI developments",
        lang="en",
        include_news=True
    )
    
    print(f"Advanced search success: {advanced_result['success']}")
    print(f"Sources used: {advanced_result.get('sources', [])}")
    
    # Show stats
    print("\n\nSearch Statistics")
    print("=" * 60)
    stats = get_search_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")