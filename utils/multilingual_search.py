"""
Multilingual Search for DocuKnow AI

Purpose:
- Free, real-time search in multiple languages
- Uses DuckDuckGo Instant Answers + Wikipedia API
- Hybrid approach for best results
- No API keys required for basic functionality
- Fallback mechanisms for reliability

Features:
1. DuckDuckGo Instant Answers (fast, real-time)
2. Wikipedia API (factual, structured)
3. Translation support for multilingual queries
4. Hybrid ranking for best results
5. Caching for performance
"""

import requests
import time
import logging
from typing import Optional, Dict, List, Tuple
import urllib.parse
import json
from datetime import datetime, timedelta
import hashlib

# Try to import translator for multilingual support
try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("Google Translate not available. Install: pip install googletrans==3.1.0a0")

logger = logging.getLogger(__name__)

class SearchResult:
    """Container for search results with metadata."""
    
    def __init__(self, content: str, source: str, confidence: float = 1.0):
        self.content = content
        self.source = source
        self.confidence = confidence
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        return f"[{self.source}] {self.content[:100]}..."

class MultilingualSearch:
    def __init__(self, timeout: int = 10, cache_size: int = 100):
        """
        Initialize multilingual search engine.
        
        Args:
            timeout: Request timeout in seconds
            cache_size: Number of search results to cache
        """
        self.timeout = timeout
        self.cache = {}
        self.cache_size = cache_size
        
        # Initialize translator if available
        self.translator = None
        if TRANSLATOR_AVAILABLE:
            try:
                self.translator = Translator()
                logger.info("Translator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize translator: {e}")
        
        # User-Agent for requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 DocuKnowAI/1.0"
        }
    
    def _get_cache_key(self, query: str, lang: str) -> str:
        """Generate cache key for query and language."""
        key_str = f"{query}_{lang}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cache_result(self, query: str, lang: str, result: SearchResult):
        """Cache search result."""
        cache_key = self._get_cache_key(query, lang)
        self.cache[cache_key] = result
        
        # Limit cache size
        if len(self.cache) > self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
    
    def _get_cached_result(self, query: str, lang: str) -> Optional[SearchResult]:
        """Get cached search result if available and not expired."""
        cache_key = self._get_cache_key(query, lang)
        
        if cache_key in self.cache:
            result = self.cache[cache_key]
            # Check if cache is still valid (5 minutes)
            if datetime.now() - result.timestamp < timedelta(minutes=5):
                return result
            else:
                # Remove expired cache
                del self.cache[cache_key]
        
        return None
    
    def duckduckgo_search(self, query: str, lang: str = "en") -> Optional[SearchResult]:
        """
        Search using DuckDuckGo Instant Answer API.
        
        Advantages:
        - Free, no API key required
        - Real-time information
        - Fast response
        - Good for current events, facts, definitions
        
        Args:
            query: Search query
            lang: Language code (affects results)
            
        Returns:
            SearchResult or None
        """
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
                "t": "docuknow_ai",
                "kl": lang  # Language region
            }
            
            logger.debug(f"DuckDuckGo search: {query} (lang: {lang})")
            
            response = requests.get(
                url, 
                params=params, 
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                result_text = None
                confidence = 0.8  # Base confidence
                
                # Check for Abstract (highest priority)
                if data.get("AbstractText"):
                    result_text = data["AbstractText"]
                    confidence = 0.9
                    logger.debug(f"DuckDuckGo: Found abstract ({len(result_text)} chars)")
                
                # Check for Definition
                elif data.get("Definition"):
                    result_text = data["Definition"]
                    confidence = 0.85
                    logger.debug(f"DuckDuckGo: Found definition")
                
                # Check for Related Topics
                elif data.get("RelatedTopics"):
                    topics = []
                    for topic in data["RelatedTopics"][:3]:  # Limit to 3
                        if isinstance(topic, dict) and "Text" in topic:
                            topics.append(topic["Text"])
                        elif isinstance(topic, str):
                            topics.append(topic)
                    
                    if topics:
                        result_text = "\n".join(topics[:3])
                        confidence = 0.7
                        logger.debug(f"DuckDuckGo: Found {len(topics)} related topics")
                
                # Check for Answer (direct answer)
                elif data.get("Answer"):
                    result_text = data["Answer"]
                    confidence = 0.95
                    logger.debug(f"DuckDuckGo: Found direct answer")
                
                if result_text:
                    # Clean up the text
                    result_text = result_text.strip()
                    
                    # Remove HTML entities if any
                    result_text = result_text.replace("&quot;", '"').replace("&amp;", "&")
                    
                    return SearchResult(result_text, "duckduckgo", confidence)
            
            logger.debug(f"DuckDuckGo: No relevant results for '{query}'")
            return None
            
        except requests.exceptions.Timeout:
            logger.warning(f"DuckDuckGo search timeout for: {query}")
            return None
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return None
    
    def wikipedia_search(self, query: str, lang: str = "en") -> Optional[SearchResult]:
        """
        Search Wikipedia for factual information.
        
        Advantages:
        - Highly accurate for factual queries
        - Well-structured information
        - Multiple language support
        - Free API
        
        Args:
            query: Search query
            lang: Wikipedia language code
            
        Returns:
            SearchResult or None
        """
        try:
            # First, search for relevant page
            search_url = f"https://{lang}.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 3,  # Get top 3 results
                "srprop": "snippet"
            }
            
            logger.debug(f"Wikipedia search: {query} (lang: {lang})")
            
            search_response = requests.get(
                search_url, 
                params=search_params, 
                headers=self.headers,
                timeout=self.timeout
            )
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                
                if search_data["query"]["search"]:
                    # Try each search result
                    for search_result in search_data["query"]["search"][:2]:  # Try first 2
                        page_id = search_result["pageid"]
                        
                        # Get page extract (introductory text)
                        extract_url = f"https://{lang}.wikipedia.org/w/api.php"
                        extract_params = {
                            "action": "query",
                            "pageids": page_id,
                            "prop": "extracts",
                            "exintro": "1",  # Only introductory section
                            "explaintext": "1",  # Plain text, no HTML
                            "format": "json",
                            "exsentences": 5  # Limit to 5 sentences
                        }
                        
                        extract_response = requests.get(
                            extract_url, 
                            params=extract_params, 
                            headers=self.headers,
                            timeout=self.timeout
                        )
                        
                        if extract_response.status_code == 200:
                            extract_data = extract_response.json()
                            pages = extract_data["query"]["pages"]
                            
                            if str(page_id) in pages and "extract" in pages[str(page_id)]:
                                extract_text = pages[str(page_id)]["extract"].strip()
                                
                                if extract_text and len(extract_text) > 50:
                                    # Calculate confidence based on snippet match
                                    snippet = search_result.get("snippet", "").lower()
                                    query_words = query.lower().split()
                                    
                                    match_score = sum(1 for word in query_words if word in snippet)
                                    confidence = 0.7 + (match_score / len(query_words) * 0.2)
                                    
                                    result = SearchResult(extract_text[:1500], "wikipedia", confidence)
                                    logger.debug(f"Wikipedia: Found article '{search_result['title']}'")
                                    return result
            
            logger.debug(f"Wikipedia: No relevant articles for '{query}'")
            return None
            
        except requests.exceptions.Timeout:
            logger.warning(f"Wikipedia search timeout for: {query}")
            return None
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return None
    
    def news_search(self, query: str, lang: str = "en") -> Optional[SearchResult]:
        """
        Search for recent news (using DuckDuckGo news).
        
        Args:
            query: News search query
            lang: Language code
            
        Returns:
            SearchResult or None
        """
        try:
            # DuckDuckGo news search (approximate)
            url = "https://duckduckgo.com/news.js"
            params = {
                "q": query,
                "kl": lang,
                "df": "d"  # Date filter (d = day)
            }
            
            response = requests.get(
                url, 
                params=params, 
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # DuckDuckGo news returns JavaScript, parse carefully
                content = response.text
                
                # Simple extraction (this is a simplified approach)
                if '"results"' in content:
                    # Extract news snippets
                    import re
                    snippets = re.findall(r'"snippet":"([^"]+)"', content)
                    
                    if snippets:
                        news_text = "\n".join(snippets[:3])  # Top 3 news snippets
                        return SearchResult(news_text, "news", 0.75)
            
            return None
            
        except Exception as e:
            logger.debug(f"News search failed: {e}")
            return None
    
    def translate_query(self, query: str, target_lang: str = "en") -> str:
        """
        Translate query to target language if needed.
        
        Args:
            query: Original query
            target_lang: Target language code
            
        Returns:
            Translated query
        """
        if not self.translator or target_lang == "en":
            return query
        
        try:
            # Detect source language
            detected = self.translator.detect(query)
            source_lang = detected.lang
            
            if source_lang != target_lang:
                translated = self.translator.translate(query, dest=target_lang, src=source_lang)
                logger.debug(f"Translated query: '{query}' ({source_lang}) -> '{translated.text}' ({target_lang})")
                return translated.text
        
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
        
        return query
    
    def translate_result(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """
        Translate search result to target language.
        
        Args:
            text: Text to translate
            target_lang: Target language
            source_lang: Source language
            
        Returns:
            Translated text
        """
        if not self.translator or target_lang == source_lang:
            return text
        
        try:
            translated = self.translator.translate(text, dest=target_lang, src=source_lang)
            return translated.text
        except Exception as e:
            logger.warning(f"Result translation failed: {e}")
            return text
    
    def hybrid_search(self, query: str, lang: str = "en") -> Optional[SearchResult]:
        """
        Try multiple search sources and return the best result.
        
        Strategy:
        1. Try DuckDuckGo (fast, real-time)
        2. Try Wikipedia (factual, structured)
        3. Try news search (for current events)
        
        Args:
            query: Search query
            lang: Language code
            
        Returns:
            Best SearchResult or None
        """
        results = []
        
        # Check cache first
        cached = self._get_cached_result(query, lang)
        if cached:
            logger.debug(f"Using cached result for: {query}")
            return cached
        
        logger.info(f"Hybrid search for: '{query}' in {lang}")
        
        # Translate query to English for better search results
        # (Most search APIs work best with English)
        translated_query = self.translate_query(query, "en")
        
        # 1. DuckDuckGo search (fastest, good for current info)
        ddg_result = self.duckduckgo_search(translated_query, "en")
        if ddg_result:
            results.append(ddg_result)
        
        # 2. Wikipedia search (best for factual queries)
        # Use original language for Wikipedia if supported
        wikipedia_lang = lang if lang in ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "hi"] else "en"
        wiki_result = self.wikipedia_search(translated_query, wikipedia_lang)
        if wiki_result:
            results.append(wiki_result)
        
        # 3. News search (for current events)
        if any(word in query.lower() for word in ["news", "latest", "today", "recent", "update"]):
            news_result = self.news_search(translated_query, "en")
            if news_result:
                results.append(news_result)
        
        # Select best result
        if results:
            # Score and rank results
            scored_results = []
            for result in results:
                score = result.confidence
                
                # Boost score based on source
                if result.source == "wikipedia" and "what" in query.lower() or "who" in query.lower():
                    score *= 1.2  # Wikipedia is better for factual queries
                
                # Boost score based on length (within reason)
                content_length = len(result.content)
                if 100 <= content_length <= 1000:
                    length_score = min(content_length / 500, 1.0)
                    score *= (0.8 + 0.2 * length_score)
                
                scored_results.append((score, result))
            
            # Sort by score (descending)
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            best_result = scored_results[0][1]
            
            # Translate result back to original language if needed
            if lang != "en" and self.translator:
                try:
                    translated_content = self.translate_result(
                        best_result.content, 
                        lang, 
                        "en"
                    )
                    best_result.content = translated_content
                    best_result.confidence *= 0.9  # Slightly reduce confidence due to translation
                except Exception as e:
                    logger.warning(f"Failed to translate result to {lang}: {e}")
            
            # Cache the result
            self._cache_result(query, lang, best_result)
            
            logger.info(f"Hybrid search successful: {best_result.source} (confidence: {best_result.confidence:.2f})")
            return best_result
        
        logger.info(f"No search results found for: '{query}'")
        return None
    
    def search(self, query: str, lang: str = "en", use_cache: bool = True) -> Optional[str]:
        """
        Main search function with language support.
        
        Args:
            query: Search query
            lang: Language code for results
            use_cache: Whether to use cache
            
        Returns:
            Search results as string or None
        """
        if not query or len(query.strip()) < 2:
            return None
        
        try:
            if not use_cache:
                # Clear cache for this query
                cache_key = self._get_cache_key(query, lang)
                if cache_key in self.cache:
                    del self.cache[cache_key]
            
            result = self.hybrid_search(query, lang)
            
            if result:
                # Format result nicely
                formatted_result = f"{result.content}\n\n[Source: {result.source.title()}]"
                
                # Add timestamp for time-sensitive info
                if any(word in query.lower() for word in ["today", "now", "current", "latest"]):
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    formatted_result = f"[Info as of {current_time}]\n\n{formatted_result}"
                
                return formatted_result[:2000]  # Limit length
            
            return None
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None
    
    def batch_search(self, queries: List[str], lang: str = "en") -> Dict[str, Optional[str]]:
        """
        Search multiple queries efficiently.
        
        Args:
            queries: List of search queries
            lang: Language code
            
        Returns:
            Dictionary mapping queries to results
        """
        results = {}
        
        for query in queries:
            results[query] = self.search(query, lang)
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def clear_cache(self):
        """Clear all cached search results."""
        self.cache.clear()
        logger.info("Search cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_max_size": self.cache_size,
            "cache_keys": list(self.cache.keys())[:10]  # First 10 keys
        }


# Global instance for easy access
_search_engine = None

def get_search_engine() -> MultilingualSearch:
    """Get or create global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = MultilingualSearch()
    return _search_engine


# Backward compatibility function
def web_search(query: str, api_key: str | None = None, lang: str = "en") -> Optional[str]:
    """
    Updated web_search function with multilingual support.
    
    Args:
        query: Search query
        api_key: Optional API key (for backward compatibility)
        lang: Language code for results
        
    Returns:
        Search results as string or None
    """
    try:
        searcher = get_search_engine()
        result = searcher.search(query, lang)
        
        if result:
            # If Tavily API key is provided, we could fall back to it
            # but our multilingual search should be sufficient
            return result
        
        # Fallback to DuckDuckGo directly if everything else fails
        if not result:
            try:
                url = "https://api.duckduckgo.com/"
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "t": "docuknow_ai"
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("AbstractText"):
                        return data["AbstractText"][:1000]
            except:
                pass
        
        return None
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return None


# Quick test function
if __name__ == "__main__":
    # Test the search engine
    logging.basicConfig(level=logging.INFO)
    
    searcher = MultilingualSearch()
    
    test_queries = [
        ("What is artificial intelligence?", "en"),
        ("आर्टिफिशियल इंटेलिजेंस क्या है?", "hi"),
        ("Qu'est-ce que l'intelligence artificielle?", "fr"),
        ("今天北京的天气怎么样？", "zh"),
    ]
    
    for query, lang in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Language: {lang}")
        print(f"{'='*60}")
        
        result = searcher.search(query, lang)
        if result:
            print(f"Result: {result[:200]}...")
        else:
            print("No result found")
        
        time.sleep(1)  # Be nice to the APIs