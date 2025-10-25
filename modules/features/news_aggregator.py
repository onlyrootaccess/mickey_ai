# Free news APIs
"""
Mickey AI - News Aggregator
Fetches and summarizes news from multiple free news APIs with intelligent categorization
"""

import logging
import requests
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import feedparser
from collections import defaultdict, Counter

class NewsCategory(Enum):
    GENERAL = "general"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    HEALTH = "health"
    SCIENCE = "science"
    POLITICS = "politics"

class NewsSource(Enum):
    NEWS_API = "newsapi"
    RSS_FEEDS = "rss"
    REDDIT = "reddit"
    GUARDIAN = "guardian"

class NewsAggregator:
    def __init__(self, newsapi_key: str = None, cache_duration: int = 900):
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.newsapi_key = newsapi_key
        self.newsapi_url = "https://newsapi.org/v2"
        
        # RSS feed sources
        self.rss_feeds = self._initialize_rss_feeds()
        
        # Cache system
        self.news_cache = {}
        self.cache_duration = cache_duration  # 15 minutes default
        
        # News processing
        self.keyword_filters = self._initialize_keyword_filters()
        self.category_keywords = self._initialize_category_keywords()
        
        # Mickey's news personalities
        self.news_messages = {
            NewsCategory.GENERAL: [
                "Mickey's got the latest news! 📰",
                "News update from Mickey! 🗞️",
                "Here's what's happening in the world! 🌍"
            ],
            NewsCategory.TECHNOLOGY: [
                "Tech news hot off the press! 💻",
                "Mickey's tech update! 🚀",
                "Latest from the tech world! 🔧"
            ],
            NewsCategory.BUSINESS: [
                "Business news from Mickey! 💼",
                "Market updates incoming! 📈",
                "Business briefing! 💰"
            ],
            NewsCategory.SPORTS: [
                "Sports news from Mickey! 🏆",
                "Game on! Sports updates! ⚽",
                "Sports highlights! 🏀"
            ],
            NewsCategory.ENTERTAINMENT: [
                "Entertainment news! 🎬",
                "Celebrity scoop from Mickey! 🌟",
                "What's hot in entertainment! 🎭"
            ],
            NewsCategory.HEALTH: [
                "Health news update! 🏥",
                "Mickey's health briefing! 💊",
                "Wellness news! 🥦"
            ],
            NewsCategory.SCIENCE: [
                "Science news from Mickey! 🔬",
                "Latest scientific discoveries! 🧪",
                "Science update! 🌌"
            ]
        }
        
        self.logger.info("📰 News Aggregator initialized - Ready to fetch headlines!")

    def _initialize_rss_feeds(self) -> Dict[NewsCategory, List[str]]:
        """Initialize RSS feed URLs for different categories"""
        return {
            NewsCategory.GENERAL: [
                "http://feeds.bbci.co.uk/news/rss.xml",
                "https://rss.cnn.com/rss/edition.rss",
                "https://feeds.reuters.com/reuters/topNews"
            ],
            NewsCategory.TECHNOLOGY: [
                "https://feeds.arstechnica.com/arstechnica/index",
                "https://www.wired.com/feed/rss",
                "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"
            ],
            NewsCategory.BUSINESS: [
                "https://feeds.reuters.com/reuters/businessNews",
                "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
                "https://www.bloomberg.com/feeds/podcasts/etf-report.xml"
            ],
            NewsCategory.SPORTS: [
                "http://feeds.bbci.co.uk/sport/rss.xml",
                "https://espn.com/espn/rss/news",
                "https://feeds.reuters.com/reuters/sportsNews"
            ],
            NewsCategory.ENTERTAINMENT: [
                "https://rss.nytimes.com/services/xml/rss/nyt/Movies.xml",
                "https://feeds.feedburner.com/people",
                "https://www.hollywoodreporter.com/rss/news"
            ],
            NewsCategory.HEALTH: [
                "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
                "https://feeds.reuters.com/reuters/healthNews",
                "https://www.npr.org/rss/rss.php?id=1128"
            ],
            NewsCategory.SCIENCE: [
                "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
                "https://feeds.reuters.com/reuters/scienceNews",
                "https://www.sciencedaily.com/rss/all.xml"
            ]
        }

    def _initialize_keyword_filters(self) -> List[str]:
        """Initialize keywords to filter out low-quality news"""
        return [
            "clickbait", "gossip", "rumor", "unconfirmed", "allegedly",
            "shocking", "you won't believe", "this will blow your mind"
        ]

    def _initialize_category_keywords(self) -> Dict[NewsCategory, List[str]]:
        """Initialize keywords for automatic news categorization"""
        return {
            NewsCategory.TECHNOLOGY: [
                "tech", "software", "ai", "artificial intelligence", "machine learning",
                "computer", "digital", "internet", "social media", "app", "startup",
                "innovation", "robot", "cybersecurity", "blockchain", "crypto"
            ],
            NewsCategory.BUSINESS: [
                "business", "market", "stock", "economy", "financial", "investment",
                "company", "corporate", "merger", "acquisition", "profit", "revenue",
                "startup", "venture capital", "trade", "commerce"
            ],
            NewsCategory.SPORTS: [
                "sports", "game", "match", "tournament", "championship", "player",
                "team", "score", "win", "loss", "football", "basketball", "soccer",
                "baseball", "tennis", "golf", "olympics", "world cup"
            ],
            NewsCategory.ENTERTAINMENT: [
                "movie", "film", "celebrity", "actor", "actress", "hollywood",
                "music", "song", "album", "concert", "award", "oscar", "grammy",
                "tv", "television", "series", "netflix", "disney"
            ],
            NewsCategory.HEALTH: [
                "health", "medical", "doctor", "hospital", "medicine", "treatment",
                "disease", "virus", "vaccine", "covid", "pandemic", "wellness",
                "fitness", "nutrition", "diet", "exercise", "mental health"
            ],
            NewsCategory.SCIENCE: [
                "science", "scientific", "research", "study", "discovery",
                "space", "nasa", "astronomy", "physics", "chemistry", "biology",
                "climate", "environment", "nature", "evolution", "quantum"
            ],
            NewsCategory.POLITICS: [
                "politics", "government", "election", "president", "congress",
                "policy", "law", "bill", "senate", "democrat", "republican",
                "international", "diplomacy", "treaty", "summit"
            ]
        }

    def set_newsapi_key(self, api_key: str):
        """Set NewsAPI key"""
        self.newsapi_key = api_key
        self.logger.info("NewsAPI key configured")

    def get_top_headlines(self, category: NewsCategory = NewsCategory.GENERAL, 
                         country: str = "us", page_size: int = 10) -> Dict[str, Any]:
        """
        Get top headlines from multiple sources
        
        Args:
            category: News category
            country: Country code for news
            page_size: Number of articles to return
            
        Returns:
            Dictionary with news headlines
        """
        try:
            cache_key = f"headlines_{category.value}_{country}_{page_size}"
            cached_data = self._get_cached_news(cache_key)
            if cached_data:
                self.logger.info("Using cached news data")
                return cached_data

            all_articles = []
            
            # Fetch from NewsAPI if available
            if self.newsapi_key:
                newsapi_articles = self._fetch_newsapi_headlines(category, country, page_size)
                all_articles.extend(newsapi_articles)
            
            # Fetch from RSS feeds
            rss_articles = self._fetch_rss_headlines(category, page_size)
            all_articles.extend(rss_articles)
            
            # Remove duplicates and filter quality
            unique_articles = self._deduplicate_articles(all_articles)
            filtered_articles = self._filter_quality_articles(unique_articles)
            
            # Sort by relevance and recency
            sorted_articles = self._sort_articles(filtered_articles)
            
            # Take requested number of articles
            final_articles = sorted_articles[:page_size]
            
            result = {
                'success': True,
                'category': category.value,
                'country': country,
                'article_count': len(final_articles),
                'articles': final_articles,
                'timestamp': datetime.now().isoformat(),
                'sources_used': self._get_used_sources(final_articles),
                'mickey_response': self._get_news_message(category, len(final_articles))
            }
            
            # Cache the result
            self._cache_news(cache_key, result)
            
            self.logger.info(f"Fetched {len(final_articles)} headlines for {category.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Headlines fetch failed: {str(e)}")
            return self._create_error_response(f"News fetch failed: {str(e)}")

    def search_news(self, query: str, category: NewsCategory = None, 
                   page_size: int = 10) -> Dict[str, Any]:
        """
        Search for news articles by query
        
        Args:
            query: Search query
            category: Optional category filter
            page_size: Number of articles to return
            
        Returns:
            Dictionary with search results
        """
        try:
            cache_key = f"search_{query}_{category.value if category else 'all'}_{page_size}"
            cached_data = self._get_cached_news(cache_key)
            if cached_data:
                self.logger.info("Using cached search results")
                return cached_data

            all_articles = []
            
            # Search RSS feeds for matching articles
            rss_articles = self._search_rss_feeds(query, category, page_size * 2)
            all_articles.extend(rss_articles)
            
            # If NewsAPI available, use it for search
            if self.newsapi_key:
                newsapi_articles = self._search_newsapi(query, category, page_size)
                all_articles.extend(newsapi_articles)
            
            # Process and filter results
            unique_articles = self._deduplicate_articles(all_articles)
            relevant_articles = self._rank_by_relevance(unique_articles, query)
            filtered_articles = self._filter_quality_articles(relevant_articles)
            
            final_articles = filtered_articles[:page_size]
            
            # Auto-detect category if not specified
            detected_category = category or self._detect_category_from_articles(final_articles)
            
            result = {
                'success': True,
                'query': query,
                'category': detected_category.value if detected_category else 'general',
                'article_count': len(final_articles),
                'articles': final_articles,
                'timestamp': datetime.now().isoformat(),
                'mickey_response': self._get_search_message(query, len(final_articles))
            }
            
            # Cache the result
            self._cache_news(cache_key, result)
            
            self.logger.info(f"Search found {len(final_articles)} articles for: {query}")
            return result
            
        except Exception as e:
            self.logger.error(f"News search failed: {str(e)}")
            return self._create_error_response(f"News search failed: {str(e)}")

    def _fetch_newsapi_headlines(self, category: NewsCategory, country: str, 
                               page_size: int) -> List[Dict[str, Any]]:
        """Fetch headlines from NewsAPI"""
        try:
            url = f"{self.newsapi_url}/top-headlines"
            params = {
                'category': category.value,
                'country': country,
                'pageSize': page_size,
                'apiKey': self.newsapi_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                processed_article = self._process_newsapi_article(article, category)
                if processed_article:
                    articles.append(processed_article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"NewsAPI fetch failed: {str(e)}")
            return []

    def _fetch_rss_headlines(self, category: NewsCategory, page_size: int) -> List[Dict[str, Any]]:
        """Fetch headlines from RSS feeds"""
        articles = []
        feed_urls = self.rss_feeds.get(category, [])
        
        for feed_url in feed_urls[:3]:  # Limit to 3 feeds per category
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:page_size // 2]:  # Take few from each feed
                    processed_article = self._process_rss_article(entry, category, feed_url)
                    if processed_article:
                        articles.append(processed_article)
                
                # Small delay to be respectful to servers
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"RSS feed failed {feed_url}: {str(e)}")
                continue
        
        return articles

    def _search_rss_feeds(self, query: str, category: NewsCategory = None, 
                         limit: int = 20) -> List[Dict[str, Any]]:
        """Search RSS feeds for query"""
        articles = []
        categories_to_search = [category] if category else list(NewsCategory)
        
        for cat in categories_to_search:
            feed_urls = self.rss_feeds.get(cat, [])
            
            for feed_url in feed_urls[:2]:  # Limit feeds per category
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries:
                        if self._article_matches_query(entry, query):
                            processed_article = self._process_rss_article(entry, cat, feed_url)
                            if processed_article:
                                articles.append(processed_article)
                                if len(articles) >= limit:
                                    return articles
                
                except Exception as e:
                    self.logger.error(f"RSS search failed {feed_url}: {str(e)}")
                    continue
        
        return articles

    def _search_newsapi(self, query: str, category: NewsCategory = None, 
                       page_size: int = 10) -> List[Dict[str, Any]]:
        """Search NewsAPI for query"""
        try:
            url = f"{self.newsapi_url}/everything"
            params = {
                'q': query,
                'pageSize': page_size,
                'sortBy': 'relevancy',
                'apiKey': self.newsapi_key
            }
            
            if category:
                params['category'] = category.value
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                processed_article = self._process_newsapi_article(
                    article, 
                    category or NewsCategory.GENERAL
                )
                if processed_article:
                    articles.append(processed_article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"NewsAPI search failed: {str(e)}")
            return []

    def _process_newsapi_article(self, article: Dict, category: NewsCategory) -> Optional[Dict[str, Any]]:
        """Process NewsAPI article into standard format"""
        try:
            # Basic validation
            if not article.get('title') or not article.get('url'):
                return None
            
            # Detect category if not provided
            detected_category = self._detect_article_category(article, category)
            
            return {
                'title': article['title'],
                'description': article.get('description', ''),
                'url': article['url'],
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'author': article.get('author', ''),
                'image_url': article.get('urlToImage', ''),
                'category': detected_category.value,
                'content': article.get('content', ''),
                'source_type': NewsSource.NEWS_API.value
            }
        except Exception as e:
            self.logger.error(f"NewsAPI article processing failed: {str(e)}")
            return None

    def _process_rss_article(self, entry: Any, category: NewsCategory, 
                           feed_url: str) -> Optional[Dict[str, Any]]:
        """Process RSS article into standard format"""
        try:
            # Basic validation
            if not entry.get('title') or not entry.get('link'):
                return None
            
            # Extract description
            description = ''
            if hasattr(entry, 'summary'):
                description = entry.summary
            elif hasattr(entry, 'description'):
                description = entry.description
            
            # Clean description (remove HTML tags)
            description = re.sub('<[^<]+?>', '', description)
            
            # Extract published date
            published_at = ''
            if hasattr(entry, 'published'):
                published_at = entry.published
            elif hasattr(entry, 'updated'):
                published_at = entry.updated
            
            # Detect category
            detected_category = self._detect_article_category(entry, category)
            
            return {
                'title': entry.title,
                'description': description[:200] + '...' if len(description) > 200 else description,
                'url': entry.link,
                'source': self._extract_source_from_feed(feed_url),
                'published_at': published_at,
                'author': getattr(entry, 'author', ''),
                'image_url': self._extract_image_from_rss(entry),
                'category': detected_category.value,
                'content': description,
                'source_type': NewsSource.RSS_FEEDS.value
            }
        except Exception as e:
            self.logger.error(f"RSS article processing failed: {str(e)}")
            return None

    def _detect_article_category(self, article: Any, default_category: NewsCategory) -> NewsCategory:
        """Detect article category from content"""
        try:
            # Combine title and description for analysis
            text = ''
            if hasattr(article, 'title'):
                text += article.title + ' '
            if hasattr(article, 'description'):
                text += getattr(article, 'description', '')
            elif isinstance(article, dict) and article.get('description'):
                text += article['description']
            
            text = text.lower()
            
            # Count keyword matches for each category
            category_scores = {}
            for category, keywords in self.category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text)
                category_scores[category] = score
            
            # Find category with highest score
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                if best_category[1] > 0:  # Only use if we found matches
                    return best_category[0]
            
            return default_category
            
        except Exception as e:
            self.logger.error(f"Category detection failed: {str(e)}")
            return default_category

    def _detect_category_from_articles(self, articles: List[Dict]) -> Optional[NewsCategory]:
        """Detect dominant category from a list of articles"""
        try:
            category_counts = Counter(article['category'] for article in articles)
            if category_counts:
                dominant_category = category_counts.most_common(1)[0][0]
                return NewsCategory(dominant_category)
            return None
        except:
            return None

    def _article_matches_query(self, article: Any, query: str) -> bool:
        """Check if article matches search query"""
        try:
            search_text = ''
            if hasattr(article, 'title'):
                search_text += article.title + ' '
            if hasattr(article, 'summary'):
                search_text += article.summary + ' '
            elif hasattr(article, 'description'):
                search_text += article.description + ' '
            
            search_text = search_text.lower()
            query_terms = query.lower().split()
            
            # Check if all query terms appear in the article
            return all(term in search_text for term in query_terms)
            
        except:
            return False

    def _extract_source_from_feed(self, feed_url: str) -> str:
        """Extract source name from feed URL"""
        try:
            # Extract domain name
            domain = feed_url.split('//')[-1].split('/')[0]
            # Remove www and common TLDs
            source = domain.replace('www.', '').split('.')[0]
            return source.title()
        except:
            return "Unknown Source"

    def _extract_image_from_rss(self, entry: Any) -> str:
        """Extract image URL from RSS entry"""
        try:
            # Check for media content
            if hasattr(entry, 'media_content') and entry.media_content:
                return entry.media_content[0]['url']
            elif hasattr(entry, 'links') and entry.links:
                for link in entry.links:
                    if link.get('type', '').startswith('image/'):
                        return link['href']
            return ''
        except:
            return ''

    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            # Create a normalized version of the title for comparison
            normalized_title = self._normalize_text(article['title'])
            
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        return unique_articles

    def _filter_quality_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filter out low-quality articles"""
        filtered_articles = []
        
        for article in articles:
            title = article['title'].lower()
            description = article.get('description', '').lower()
            
            # Check for clickbait patterns
            is_clickbait = any(pattern in title or pattern in description 
                              for pattern in self.keyword_filters)
            
            # Check for sufficient content
            has_content = len(article['title']) > 10 and len(description) > 20
            
            if not is_clickbait and has_content:
                filtered_articles.append(article)
        
        return filtered_articles

    def _sort_articles(self, articles: List[Dict]) -> List[Dict]:
        """Sort articles by relevance and recency"""
        def article_score(article):
            score = 0
            
            # Prefer articles with images
            if article.get('image_url'):
                score += 10
            
            # Prefer articles from reputable sources
            source = article.get('source', '').lower()
            if any(reputable in source for reputable in ['bbc', 'reuters', 'associated press']):
                score += 5
            
            # Prefer recent articles (if we have dates)
            if article.get('published_at'):
                try:
                    # Simple recency scoring - newer is better
                    published_time = self._parse_date(article['published_at'])
                    if published_time:
                        days_ago = (datetime.now() - published_time).days
                        score += max(0, 10 - days_ago)  # Up to 10 points for recency
                except:
                    pass
            
            return score
        
        return sorted(articles, key=article_score, reverse=True)

    def _rank_by_relevance(self, articles: List[Dict], query: str) -> List[Dict]:
        """Rank articles by relevance to search query"""
        query_terms = query.lower().split()
        
        def relevance_score(article):
            score = 0
            text = f"{article['title']} {article.get('description', '')}".lower()
            
            # Count exact matches
            for term in query_terms:
                score += text.count(term) * 2
            
            # Bonus for title matches
            title = article['title'].lower()
            for term in query_terms:
                if term in title:
                    score += 3
            
            return score
        
        return sorted(articles, key=relevance_score, reverse=True)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats"""
        try:
            # Try common RSS date formats
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%a, %d %b %Y %H:%M:%S %z',
                       '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
            return None
        except:
            return None

    def _get_used_sources(self, articles: List[Dict]) -> List[str]:
        """Get list of sources used in articles"""
        sources = set(article['source'] for article in articles if article.get('source'))
        return list(sources)

    def _get_news_message(self, category: NewsCategory, article_count: int) -> str:
        """Get Mickey's news message"""
        import random
        messages = self.news_messages.get(category, ["News update from Mickey! 📰"])
        base_message = random.choice(messages)
        return f"{base_message} Found {article_count} articles!"

    def _get_search_message(self, query: str, article_count: int) -> str:
        """Get Mickey's search message"""
        if article_count == 0:
            return f"Mickey couldn't find any news about '{query}'! 🤔"
        else:
            return f"Mickey found {article_count} articles about '{query}'! 📰"

    def _get_cached_news(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached news data if valid"""
        if cache_key in self.news_cache:
            cached_time, data = self.news_cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return data
            else:
                del self.news_cache[cache_key]
        return None

    def _cache_news(self, cache_key: str, data: Dict[str, Any]):
        """Cache news data"""
        self.news_cache[cache_key] = (time.time(), data)
        
        # Clean up old cache
        current_time = time.time()
        self.news_cache = {
            k: v for k, v in self.news_cache.items() 
            if current_time - v[0] < self.cache_duration * 2
        }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'mickey_response': random.choice([
                "Oops! Mickey can't fetch news right now! 📰",
                "News service is taking a break! Try again soon! 🌟",
                "Mickey's news magic isn't working! 😅"
            ])
        }

    def get_news_categories(self) -> Dict[str, Any]:
        """Get available news categories"""
        return {
            'categories': [category.value for category in NewsCategory],
            'rss_feeds_count': sum(len(feeds) for feeds in self.rss_feeds.values()),
            'newsapi_configured': self.newsapi_key is not None
        }

    def set_cache_duration(self, duration: int):
        """Set cache duration in seconds"""
        self.cache_duration = duration
        self.logger.info(f"News cache duration set to {duration} seconds")

    def clear_cache(self):
        """Clear news cache"""
        self.news_cache.clear()
        self.logger.info("News cache cleared")

    def get_news_stats(self) -> Dict[str, Any]:
        """Get news aggregator statistics"""
        return {
            'cache_size': len(self.news_cache),
            'cache_duration_seconds': self.cache_duration,
            'newsapi_configured': self.newsapi_key is not None,
            'rss_feeds_by_category': {
                category.value: len(feeds) 
                for category, feeds in self.rss_feeds.items()
            },
            'supported_categories': [category.value for category in NewsCategory]
        }

# Test function
def test_news_aggregator():
    """Test the news aggregator"""
    news = NewsAggregator()  # No API key for testing
    
    print("Testing News Aggregator...")
    
    # Test top headlines
    headlines = news.get_top_headlines(NewsCategory.TECHNOLOGY, page_size=5)
    print("Headlines:", headlines.get('mickey_response', 'Error'))
    
    if headlines['success']:
        for article in headlines['articles'][:3]:
            print(f"- {article['title'][:50]}...")
    
    # Test search
    search = news.search_news("artificial intelligence", page_size=3)
    print("Search:", search.get('mickey_response', 'Error'))
    
    if search['success']:
        for article in search['articles']:
            print(f"- {article['title'][:50]}...")
    
    # Test categories
    categories = news.get_news_categories()
    print("Categories:", categories)
    
    # Test stats
    stats = news.get_news_stats()
    print("News Stats:", stats)

if __name__ == "__main__":
    test_news_aggregator()