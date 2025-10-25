# DuckDuckGo integration
"""
Mickey AI - Web Search Module
Provides web search capabilities using multiple search providers
"""

import logging
import requests
import json
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import urllib.parse

class SearchProvider:
    """Base class for search providers"""
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)

    def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        raise NotImplementedError

class GoogleSearchProvider(SearchProvider):
    """Google Search provider using Custom Search API"""
    def __init__(self, api_key: str, search_engine_id: str):
        super().__init__("Google")
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform Google search using Custom Search API"""
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(max_results, 10)  # Google API max is 10
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = self._parse_google_results(data)
            
            return {
                'success': True,
                'provider': self.name,
                'query': query,
                'results': results,
                'total_results': data.get('searchInformation', {}).get('totalResults', '0')
            }

        except Exception as e:
            self.logger.error(f"Google search failed: {str(e)}")
            return {
                'success': False,
                'provider': self.name,
                'error': str(e)
            }

    def _parse_google_results(self, data: Dict) -> List[Dict[str, str]]:
        """Parse Google API response"""
        results = []
        for item in data.get('items', []):
            results.append({
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'description': item.get('snippet', ''),
                'source': 'Google'
            })
        return results

class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider using their API"""
    def __init__(self):
        super().__init__("DuckDuckGo")
        self.base_url = "https://api.duckduckgo.com/"

    def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform DuckDuckGo search"""
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = self._parse_ddg_results(data, query)
            
            return {
                'success': True,
                'provider': self.name,
                'query': query,
                'results': results[:max_results],
                'total_results': str(len(results))
            }

        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {str(e)}")
            return {
                'success': False,
                'provider': self.name,
                'error': str(e)
            }

    def _parse_ddg_results(self, data: Dict, query: str) -> List[Dict[str, str]]:
        """Parse DuckDuckGo API response"""
        results = []
        
        # Add instant answer if available
        if data.get('AbstractText'):
            results.append({
                'title': data.get('Heading', 'Instant Answer'),
                'url': data.get('AbstractURL', ''),
                'description': data.get('AbstractText', ''),
                'source': 'DuckDuckGo Instant Answer'
            })
        
        # Add related topics
        for topic in data.get('RelatedTopics', []):
            if 'Text' in topic and 'FirstURL' in topic:
                results.append({
                    'title': topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                    'url': topic.get('FirstURL', ''),
                    'description': topic.get('Text', ''),
                    'source': 'DuckDuckGo Related Topic'
                })
        
        return results

class NewsSearchProvider(SearchProvider):
    """News search provider"""
    def __init__(self, news_api_key: str = None):
        super().__init__("News")
        self.news_api_key = news_api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform news search"""
        if not self.news_api_key:
            return {
                'success': False,
                'provider': self.name,
                'error': 'News API key not configured'
            }

        try:
            # Calculate date for recent news (last 7 days)
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'pageSize': min(max_results, 50),
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en'
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = self._parse_news_results(data)
            
            return {
                'success': True,
                'provider': self.name,
                'query': query,
                'results': results,
                'total_results': data.get('totalResults', 0)
            }

        except Exception as e:
            self.logger.error(f"News search failed: {str(e)}")
            return {
                'success': False,
                'provider': self.name,
                'error': str(e)
            }

    def _parse_news_results(self, data: Dict) -> List[Dict[str, str]]:
        """Parse news API response"""
        results = []
        for article in data.get('articles', []):
            results.append({
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'description': article.get('description', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'image_url': article.get('urlToImage', '')
            })
        return results

class WebSearch:
    """Main web search class that orchestrates multiple search providers"""
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.providers = []
        self.search_history = []
        self.max_history_size = 100
        
        # Initialize providers based on config
        self._initialize_providers()
        
        # Mickey's search personalities
        self.search_responses = [
            "Mickey's on the case! Searching the web for you! 🔍",
            "Hot dog! Let me find that information for you! 🌭",
            "Searching high and low! Mickey's got this! 🐭",
            "Web search activated! Gathering information! 🌐",
            "Mickey's magic search is working! ✨"
        ]
        
        self.logger.info("🔍 Web Search initialized - Ready to find anything!")

    def _initialize_providers(self):
        """Initialize search providers based on configuration"""
        # Google Custom Search
        google_api_key = self.config.get('google_api_key')
        google_engine_id = self.config.get('google_engine_id')
        if google_api_key and google_engine_id:
            self.providers.append(GoogleSearchProvider(google_api_key, google_engine_id))
        
        # DuckDuckGo (always available)
        self.providers.append(DuckDuckGoProvider())
        
        # News API
        news_api_key = self.config.get('news_api_key')
        if news_api_key:
            self.providers.append(NewsSearchProvider(news_api_key))
        
        self.logger.info(f"Initialized {len(self.providers)} search providers")

    async def search(self, query: str, search_type: str = "web", max_results: int = 10) -> Dict[str, Any]:
        """
        Perform web search across multiple providers
        
        Args:
            query: Search query
            search_type: Type of search (web, news, images, etc.)
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            self.logger.info(f"Performing {search_type} search: '{query}'")
            
            # Add to search history
            self._add_to_history(query, search_type)
            
            # Select appropriate providers based on search type
            if search_type == "news":
                providers = [p for p in self.providers if p.name == "News"]
                if not providers:
                    return self._create_error_response("News search not available")
            else:
                providers = [p for p in self.providers if p.name != "News"]
            
            # Execute searches in parallel (simulated with sequential for now)
            all_results = []
            for provider in providers:
                result = provider.search(query, max_results)
                if result.get('success'):
                    all_results.extend(result.get('results', []))
            
            # Remove duplicates based on URL
            unique_results = self._remove_duplicates(all_results)
            
            # Sort by relevance (simple heuristic)
            sorted_results = self._sort_results(unique_results, query)
            
            # Limit results
            final_results = sorted_results[:max_results]
            
            response = {
                'success': True,
                'query': query,
                'search_type': search_type,
                'results': final_results,
                'total_found': len(final_results),
                'providers_used': [p.name for p in providers],
                'mickey_response': random.choice(self.search_responses),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Search completed: Found {len(final_results)} results")
            return response
            
        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            return self._create_error_response(f"Search failed: {str(e)}")

    async def search_news(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Convenience method for news search"""
        return await self.search(query, "news", max_results)

    async def search_images(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search for images (placeholder for future implementation)"""
        # Note: This would require additional APIs or web scraping
        return {
            'success': False,
            'query': query,
            'search_type': 'images',
            'error': 'Image search not yet implemented',
            'mickey_response': "Mickey's still learning image search! Coming soon! 🖼️"
        }

    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate search results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results

    def _sort_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Sort results by relevance to query"""
        query_words = set(query.lower().split())
        
        def relevance_score(result):
            score = 0
            title = result.get('title', '').lower()
            description = result.get('description', '').lower()
            
            # Title matches are most important
            for word in query_words:
                if word in title:
                    score += 3
                if word in description:
                    score += 1
            
            # Boost results from preferred sources
            source = result.get('source', '').lower()
            if any(preferred in source for preferred in ['google', 'news']):
                score += 2
            
            return score
        
        return sorted(results, key=relevance_score, reverse=True)

    def _add_to_history(self, query: str, search_type: str):
        """Add search to history"""
        search_entry = {
            'query': query,
            'type': search_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.search_history.append(search_entry)
        
        # Limit history size
        if len(self.search_history) > self.max_history_size:
            self.search_history = self.search_history[-self.max_history_size:]

    def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Get recent search history"""
        return self.search_history[-limit:] if limit else self.search_history

    def clear_search_history(self):
        """Clear search history"""
        self.search_history = []
        self.logger.info("Search history cleared")

    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions for partial query"""
        # Simple implementation - in production, this would use a search suggestions API
        common_searches = [
            "weather today",
            "latest news",
            "cricket scores",
            "movie reviews",
            "recipe for",
            "how to",
            "what is",
            "best places to visit"
        ]
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        for common in common_searches:
            if common.startswith(partial_lower) and common != partial_lower:
                suggestions.append(common)
        
        return suggestions[:5]  # Return top 5 suggestions

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'mickey_response': random.choice([
                "Oops! Mickey couldn't find anything for that search! 😅",
                "Hot dog! The search didn't work this time! 🌭",
                "Mickey's search magic is taking a break! Try again?",
                "Uh oh! The web seems to be hiding from Mickey! 🌐"
            ])
        }

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'total_providers': len(self.providers),
            'provider_names': [p.name for p in self.providers],
            'search_history_count': len(self.search_history),
            'last_search': self.search_history[-1] if self.search_history else None
        }

# Test function
async def test_web_search():
    """Test the web search module"""
    # Note: This test will only work if API keys are configured
    config = {
        'google_api_key': 'test_key',  # Replace with actual key for testing
        'google_engine_id': 'test_engine_id',
        'news_api_key': 'test_news_key'
    }
    
    web_search = WebSearch(config)
    
    # Test web search
    result = await web_search.search("Mickey Mouse", max_results=5)
    print("Web Search Result:", json.dumps(result, indent=2, default=str))
    
    # Test search history
    history = web_search.get_search_history()
    print("Search History:", history)
    
    # Test search suggestions
    suggestions = web_search.get_search_suggestions("how to")
    print("Search Suggestions:", suggestions)
    
    # Test stats
    stats = web_search.get_search_stats()
    print("Search Stats:", stats)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_web_search())