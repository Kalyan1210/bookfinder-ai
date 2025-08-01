import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import re
import sqlite3
import json
import time
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import random
from typing import List, Dict, Any
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Configure page
st.set_page_config(
    page_title="📚 BookFinder AI - Dynamic Search Edition",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .book-card {
        border: 2px solid #ddd;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .emotion-tag {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 3px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .link-button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 8px 16px;
        text-decoration: none;
        border-radius: 25px;
        margin: 3px;
        display: inline-block;
        font-size: 0.9rem;
        transition: transform 0.2s;
    }
    .link-button:hover {
        transform: translateY(-2px);
    }
    .score-badge {
        background: linear-gradient(45deg, #ff6b6b, #ffa500);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .search-status {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .dynamic-search-indicator {
        background: linear-gradient(45deg, #ff9a9e, #fecfef);
        padding: 8px 16px;
        border-radius: 20px;
        color: #333;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

class DynamicBookSearchEngine:
    """Real-time Google Books API search engine"""
    
    def __init__(self):
        self.google_books_api = "https://www.googleapis.com/books/v1/volumes"
        self.cache_db = "dynamic_books_cache.db"
        self.init_cache_db()
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    def init_cache_db(self):
        """Initialize SQLite cache database for dynamic searches"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_cache (
                query_hash TEXT PRIMARY KEY,
                search_terms TEXT,
                results TEXT,
                timestamp REAL,
                result_count INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def generate_search_queries(self, user_input: str, recent_books: List[str]) -> List[str]:
        """Generate intelligent search queries based on user input"""
        
        # Extract key terms from user input
        blob = TextBlob(user_input.lower())
        
        # Subject mapping for intelligent query generation
        subject_patterns = {
            'data_science': ['data science', 'machine learning', 'artificial intelligence', 'python data', 'statistics programming'],
            'programming': ['programming', 'coding', 'software development', 'computer science', 'algorithms'],
            'business': ['business', 'entrepreneurship', 'startup', 'management', 'leadership'],
            'psychology': ['psychology', 'mental health', 'cognitive science', 'behavioral economics'],
            'history': ['history', 'historical', 'biography', 'world history', 'ancient'],
            'science': ['physics', 'chemistry', 'biology', 'astronomy', 'quantum'],
            'philosophy': ['philosophy', 'ethics', 'meaning', 'existence', 'wisdom'],
            'fiction': ['novel', 'story', 'fiction', 'narrative', 'literature'],
            'self_help': ['self help', 'personal development', 'motivation', 'productivity', 'habits'],
            'finance': ['finance', 'investing', 'money', 'wealth', 'economics'],
        }
        
        # Mood-based query modifiers
        mood_modifiers = {
            'inspiring': ['inspiring', 'motivational', 'uplifting'],
            'learning': ['beginner', 'introduction', 'guide', 'handbook'],
            'advanced': ['advanced', 'expert', 'mastery', 'deep dive'],
            'practical': ['practical', 'hands-on', 'applied', 'real-world'],
            'theoretical': ['theory', 'principles', 'foundations', 'concepts']
        }
        
        # Generate base queries from detected subjects
        queries = []
        user_text = user_input.lower()
        
        # Direct keyword extraction
        direct_keywords = []
        for subject, keywords in subject_patterns.items():
            for keyword in keywords:
                if keyword in user_text:
                    direct_keywords.extend(keywords[:3])  # Take top 3 related terms
        
        # If specific subjects detected, prioritize them
        if direct_keywords:
            queries.extend(direct_keywords[:5])
        
        # Add mood-modified queries
        for mood, modifiers in mood_modifiers.items():
            if any(mod in user_text for mod in modifiers):
                if direct_keywords:
                    queries.extend([f"{keyword} {mood}" for keyword in direct_keywords[:2]])
        
        # Add emotion-based queries
        emotion_keywords = self.extract_emotional_keywords(user_text)
        queries.extend(emotion_keywords[:3])
        
        # Add general fallback queries if nothing specific detected
        if not queries:
            general_terms = ['bestseller', 'popular fiction', 'non-fiction', 'highly rated']
            queries.extend(general_terms)
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(queries))[:8]
        
        return unique_queries
    
    def extract_emotional_keywords(self, text: str) -> List[str]:
        """Extract emotional context for book searches"""
        emotion_mappings = {
            'happy': ['uplifting fiction', 'comedy', 'feel good books'],
            'sad': ['emotional fiction', 'drama', 'touching stories'],
            'excited': ['adventure', 'thriller', 'action'],
            'calm': ['peaceful reads', 'meditation', 'mindfulness'],
            'curious': ['science', 'learning', 'discovery'],
            'motivated': ['self help', 'productivity', 'success'],
            'escape': ['fantasy', 'science fiction', 'adventure fiction'],
            'growth': ['personal development', 'psychology', 'philosophy']
        }
        
        emotional_queries = []
        for emotion, book_types in emotion_mappings.items():
            if emotion in text or any(keyword in text for keyword in ['feel', 'want', 'need', 'looking']):
                emotional_queries.extend(book_types[:2])
        
        return emotional_queries
    
    def search_google_books_realtime(self, query: str, max_results: int = 20) -> List[Dict]:
        """Perform real-time Google Books search"""
        
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cached_result = self.get_cached_search(query_hash)
        if cached_result:
            return cached_result
        
        try:
            # Build search URL with various parameters for better results
            search_params = {
                'q': query,
                'maxResults': max_results,
                'orderBy': 'relevance',
                'printType': 'books',
                'projection': 'full'
            }
            
            response = requests.get(self.google_books_api, params=search_params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                books = []
                
                for item in data.get('items', []):
                    book_info = self.parse_google_book_advanced(item)
                    if book_info:
                        books.append(book_info)
                
                # Cache the results
                self.cache_search_results(query_hash, query, books)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                return books
            else:
                st.warning(f"Google Books API returned status code: {response.status_code}")
                return []
                
        except Exception as e:
            st.error(f"Error searching Google Books for '{query}': {str(e)}")
            return []
    
    def get_cached_search(self, query_hash: str, max_age_hours: int = 24) -> List[Dict]:
        """Get cached search results"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT results, timestamp FROM search_cache WHERE query_hash = ?", 
                (query_hash,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                results_json, timestamp = result
                if time.time() - timestamp < max_age_hours * 3600:
                    return json.loads(results_json)
            return None
        except:
            return None
    
    def cache_search_results(self, query_hash: str, search_terms: str, results: List[Dict]):
        """Cache search results"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT OR REPLACE INTO search_cache (query_hash, search_terms, results, timestamp, result_count) VALUES (?, ?, ?, ?, ?)",
                (query_hash, search_terms, json.dumps(results), time.time(), len(results))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Cache error: {str(e)}")
    
    def parse_google_book_advanced(self, item: Dict) -> Dict:
        """Advanced parsing of Google Books API response"""
        try:
            volume_info = item.get('volumeInfo', {})
            
            # Basic info
            title = volume_info.get('title', 'Unknown Title')
            authors = volume_info.get('authors', ['Unknown Author'])
            author = ', '.join(authors) if isinstance(authors, list) else str(authors)
            
            # Skip if title or author is missing
            if title == 'Unknown Title' or not title.strip():
                return None
            
            # Categories and classification
            categories = volume_info.get('categories', ['General'])
            category = categories[0] if categories else 'General'
            
            # Enhanced genre classification
            genre = self.classify_genre_advanced(category, title, volume_info.get('description', ''))
            
            # Rating and publication info
            rating = volume_info.get('averageRating')
            if rating is None:
                # Estimate rating based on review count and other factors
                review_count = volume_info.get('ratingsCount', 0)
                rating = self.estimate_rating(review_count, category)
            
            published_date = volume_info.get('publishedDate', '2000')
            year = self.extract_year(published_date)
            
            # Description processing
            description = volume_info.get('description', '')
            if len(description) > 400:
                description = description[:400] + "..."
            
            # Enhanced emotion and keyword extraction
            emotions = self.extract_emotions_advanced(description, title, category)
            keywords = self.extract_keywords_advanced(description, title, category)
            
            # Cover image with fallbacks
            image_links = volume_info.get('imageLinks', {})
            cover_url = self.get_best_cover_image(image_links)
            
            # Additional metadata
            page_count = volume_info.get('pageCount', 0)
            language = volume_info.get('language', 'en')
            publisher = volume_info.get('publisher', 'Unknown')
            
            # Quality score (for filtering low-quality results)
            quality_score = self.calculate_quality_score(volume_info)
            
            return {
                'title': title,
                'author': author,
                'category': category,
                'genre': genre,
                'rating': round(rating, 1),
                'year': year,
                'description': description,
                'emotions': emotions,
                'keywords': keywords,
                'cover_url': cover_url,
                'google_id': item.get('id', ''),
                'page_count': page_count,
                'language': language,
                'publisher': publisher,
                'quality_score': quality_score
            }
            
        except Exception as e:
            return None
    
    def classify_genre_advanced(self, category: str, title: str, description: str) -> str:
        """Advanced genre classification"""
        combined_text = f"{category} {title} {description}".lower()
        
        genre_patterns = {
            'Data Science': ['data science', 'machine learning', 'artificial intelligence', 'python data', 'analytics', 'big data'],
            'Programming': ['programming', 'coding', 'software', 'javascript', 'python', 'java', 'algorithm'],
            'Business': ['business', 'entrepreneur', 'startup', 'management', 'leadership', 'strategy'],
            'Self-Help': ['self help', 'personal development', 'productivity', 'habits', 'success'],
            'Psychology': ['psychology', 'cognitive', 'behavioral', 'mental health', 'therapy'],
            'Science Fiction': ['science fiction', 'sci-fi', 'space', 'future', 'alien', 'dystopian'],
            'Mystery/Thriller': ['mystery', 'thriller', 'detective', 'crime', 'suspense', 'murder'],
            'Romance': ['romance', 'love story', 'romantic', 'relationship'],
            'Fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'medieval', 'mythical'],
            'Biography': ['biography', 'memoir', 'autobiography', 'life story'],
            'History': ['history', 'historical', 'war', 'ancient', 'civilization'],
            'Finance': ['finance', 'investing', 'money', 'wealth', 'economics', 'trading'],
            'Health & Fitness': ['health', 'fitness', 'diet', 'nutrition', 'exercise', 'wellness'],
            'Travel': ['travel', 'journey', 'adventure', 'guide', 'exploration'],
            'Philosophy': ['philosophy', 'philosophical', 'ethics', 'meaning', 'wisdom']
        }
        
        for genre, keywords in genre_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                return genre
        
        return category if category else 'General'
    
    def estimate_rating(self, review_count: int, category: str) -> float:
        """Estimate rating based on available data"""
        base_rating = 4.0
        
        if review_count > 1000:
            base_rating += 0.3
        elif review_count > 100:
            base_rating += 0.1
        elif review_count < 10:
            base_rating -= 0.2
        
        # Category-based adjustments
        high_rated_categories = ['self-help', 'business', 'science']
        if any(cat in category.lower() for cat in high_rated_categories):
            base_rating += 0.1
        
        return min(5.0, max(1.0, base_rating))
    
    def extract_year(self, published_date: str) -> int:
        """Extract publication year from various date formats"""
        year_patterns = [r'(\d{4})', r'(\d{4})-\d{2}', r'(\d{4})-\d{2}-\d{2}']
        
        for pattern in year_patterns:
            match = re.search(pattern, published_date)
            if match:
                year = int(match.group(1))
                if 1800 <= year <= datetime.now().year + 1:
                    return year
        
        return 2000  # Default fallback
    
    def extract_emotions_advanced(self, description: str, title: str, category: str) -> List[str]:
        """Advanced emotion extraction"""
        combined_text = f"{title} {description}".lower()
        
        emotion_patterns = {
            'inspiring': ['inspiring', 'motivational', 'uplifting', 'empowering', 'transformative'],
            'educational': ['learning', 'guide', 'handbook', 'tutorial', 'educational'],
            'practical': ['practical', 'hands-on', 'applied', 'real-world', 'actionable'],
            'entertaining': ['entertaining', 'funny', 'humorous', 'engaging', 'captivating'],
            'thought-provoking': ['thought-provoking', 'philosophical', 'deep', 'profound'],
            'emotional': ['emotional', 'touching', 'heartwarming', 'moving'],
            'thrilling': ['thrilling', 'suspenseful', 'exciting', 'gripping'],
            'romantic': ['romantic', 'love', 'passion', 'relationship'],
            'dark': ['dark', 'gritty', 'serious', 'intense'],
            'adventurous': ['adventure', 'journey', 'exploration', 'quest']
        }
        
        detected_emotions = []
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                detected_emotions.append(emotion)
        
        # Sentiment analysis fallback
        if not detected_emotions:
            try:
                blob = TextBlob(description)
                if blob.sentiment.polarity > 0.3:
                    detected_emotions.append('positive')
                elif blob.sentiment.polarity < -0.3:
                    detected_emotions.append('serious')
                else:
                    detected_emotions.append('balanced')
            except:
                detected_emotions.append('engaging')
        
        return detected_emotions[:4]
    
    def extract_keywords_advanced(self, description: str, title: str, category: str) -> str:
        """Advanced keyword extraction"""
        combined_text = f"{title} {description}".lower()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those', 'book', 'author'
        }
        
        # Extract meaningful words
        words = re.findall(r'\b\w+\b', combined_text)
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Get frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and relevance
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        return ', '.join([word for word, freq in top_keywords])
    
    def get_best_cover_image(self, image_links: Dict) -> str:
        """Get the best available cover image"""
        preference_order = ['large', 'medium', 'thumbnail', 'smallThumbnail']
        
        for size in preference_order:
            if size in image_links:
                url = image_links[size]
                if url.startswith('http:'):
                    url = url.replace('http:', 'https:')
                return url
        
        return ''
    
    def calculate_quality_score(self, volume_info: Dict) -> float:
        """Calculate book quality score for filtering"""
        score = 0.0
        
        # Has description
        if volume_info.get('description'):
            score += 2.0
        
        # Has rating
        if volume_info.get('averageRating'):
            score += 1.0
        
        # Has cover image
        if volume_info.get('imageLinks'):
            score += 1.0
        
        # Has page count
        if volume_info.get('pageCount', 0) > 50:
            score += 1.0
        
        # Recent publication (last 30 years)
        published_date = volume_info.get('publishedDate', '')
        if published_date:
            year = self.extract_year(published_date)
            if year > 1994:
                score += 1.0
        
        return score
    
    def parallel_search_multiple_queries(self, queries: List[str], books_per_query: int = 15) -> List[Dict]:
        """Search multiple queries in parallel for faster results"""
        all_books = []
        
        # Use ThreadPoolExecutor for parallel searches
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_query = {
                executor.submit(self.search_google_books_realtime, query, books_per_query): query 
                for query in queries[:6]  # Limit to 6 parallel searches
            }
            
            for future in future_to_query:
                try:
                    books = future.result(timeout=10)
                    all_books.extend(books)
                except Exception as e:
                    query = future_to_query[future]
                    st.warning(f"Search failed for '{query}': {str(e)}")
        
        # Remove duplicates based on title and author
        seen = set()
        unique_books = []
        for book in all_books:
            identifier = f"{book['title']}_{book['author']}"
            if identifier not in seen:
                seen.add(identifier)
                unique_books.append(book)
        
        # Filter by quality score
        quality_books = [book for book in unique_books if book.get('quality_score', 0) >= 3.0]
        
        return quality_books

class DynamicRecommendationEngine:
    """Enhanced recommendation engine for dynamic searches"""
    
    def __init__(self):
        self.search_engine = DynamicBookSearchEngine()
    
    def get_dynamic_recommendations(self, user_input: str, recent_books: List[str], 
                                 filters: Dict, num_recommendations: int = 5) -> List[Dict]:
        """Get recommendations through dynamic real-time search"""
        
        # Generate intelligent search queries
        search_queries = self.search_engine.generate_search_queries(user_input, recent_books)
        
        # Display search strategy to user
        st.markdown("### 🔍 Dynamic Search Strategy")
        st.markdown('<div class="search-status">🚀 Performing real-time searches on Google Books API...</div>', unsafe_allow_html=True)
        
        search_display = " ".join([f'<span class="dynamic-search-indicator">{query}</span>' for query in search_queries[:5]])
        st.markdown(f"**Search Queries:** {search_display}", unsafe_allow_html=True)
        
        # Perform parallel searches
        with st.spinner("🔄 Searching Google Books in real-time..."):
            books = self.search_engine.parallel_search_multiple_queries(search_queries, books_per_query=20)
        
        if not books:
            st.error("❌ No books found with current search terms. Try different keywords.")
            return []
        
        st.success(f"✅ Found {len(books)} books from real-time search!")
        
        # Apply filters
        filtered_books = self.apply_filters(books, filters)
        
        # Calculate recommendation scores
        scored_books = self.score_books_for_user(filtered_books, user_input, recent_books)
        
        # Return top recommendations
        return scored_books[:num_recommendations]
    
    def apply_filters(self, books: List[Dict], filters: Dict) -> List[Dict]:
        """Apply user filters to book list"""
        filtered = books.copy()
        
        # Category filter
        if filters.get('category') and filters['category'] != 'All':
            filtered = [book for book in filtered if book['category'] == filters['category']]
        
        # Genre filter
        if filters.get('genres'):
            filtered = [book for book in filtered if book['genre'] in filters['genres']]
        
        # Rating filter
        if filters.get('min_rating'):
            filtered = [book for book in filtered if book['rating'] >= filters['min_rating']]
        
        # Year range filter
        if filters.get('year_range'):
            year_min, year_max = filters['year_range']
            filtered = [book for book in filtered if year_min <= book['year'] <= year_max]
        
        # Page count filter
        if filters.get('page_range'):
            page_min, page_max = filters['page_range']
            filtered = [book for book in filtered if page_min <= book['page_count'] <= page_max]
        
        # Language filter
        if filters.get('language') and filters['language'] != 'All':
            filtered = [book for book in filtered if book.get('language', 'en') == filters['language']]
        
        return filtered
    
    def score_books_for_user(self, books: List[Dict], user_input: str, recent_books: List[str]) -> List[Dict]:
        """Score books based on user preferences"""
        
        # Analyze user preferences
        user_prefs = self.analyze_user_text(user_input)
        
        scored_books = []
        
        for book in books:
            score = self.calculate_relevance_score(book, user_prefs, recent_books)
            scored_books.append((book, score))
        
        # Sort by score descending
        scored_books.sort(key=lambda x: x[1], reverse=True)
        
        return [book for book, score in scored_books]
    
    def analyze_user_text(self, user_input: str) -> Dict:
        """Analyze user input for preferences"""
        try:
            blob = TextBlob(user_input.lower())
            
            # Extract key themes and emotions
            themes = []
            emotions = []
            
            # Theme detection
            theme_keywords = {
                'learning': ['learn', 'study', 'understand', 'education', 'knowledge'],
                'career': ['career', 'job', 'work', 'professional', 'business'],
                'personal_growth': ['grow', 'improve', 'develop', 'better', 'change'],
                'entertainment': ['fun', 'entertaining', 'enjoy', 'escape', 'relax'],
                'inspiration': ['inspire', 'motivate', 'encourage', 'uplift'],
                'practical': ['practical', 'useful', 'apply', 'real', 'hands-on']
            }
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in user_input.lower() for keyword in keywords):
                    themes.append(theme)
            
            # Emotion detection
            emotion_keywords = {
                'excited': ['excited', 'enthusiastic', 'eager'],
                'curious': ['curious', 'interested', 'wondering'],
                'focused': ['focused', 'serious', 'dedicated'],
                'relaxed': ['relaxed', 'calm', 'peaceful'],
                'ambitious': ['ambitious', 'driven', 'goal']
            }
            
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in user_input.lower() for keyword in keywords):
                    emotions.append(emotion)
            
            return {
                'themes': themes,
                'emotions': emotions,
                'sentiment': blob.sentiment.polarity,
                'text': user_input.lower()
            }
            
        except:
            return {
                'themes': ['general'],
                'emotions': ['interested'],
                'sentiment': 0.0,
                'text': user_input.lower()
            }
    
    def calculate_relevance_score(self, book: Dict, user_prefs: Dict, recent_books: List[str]) -> float:
        """Calculate how relevant a book is to user preferences"""
        score = 0.0
        
        # Base rating score (0-50 points)
        score += book['rating'] * 10
        
        # Title/description relevance (0-30 points)
        user_text = user_prefs['text']
        book_text = f"{book['title']} {book['description']} {book['keywords']}".lower()
        
        # Simple keyword matching
        user_words = set(user_text.split())
        book_words = set(book_text.split())
        common_words = user_words & book_words
        
        if len(user_words) > 0:
            relevance_ratio = len(common_words) / len(user_words)
            score += relevance_ratio * 30
        
        # Theme matching (0-20 points)
        book_themes = book.get('emotions', [])
        user_themes = user_prefs.get('themes', [])
        theme_matches = len(set(book_themes) & set(user_themes))
        score += theme_matches * 10
        
        # Quality score bonus (0-10 points)
        score += book.get('quality_score', 0) * 2
        
        # Recency penalty for older books
        current_year = datetime.now().year
        age = current_year - book['year']
        if age > 20:
            score -= age * 0.5
        
        # Avoid recently read books
        if any(recent.lower() in book['title'].lower() for recent in recent_books):
            score -= 30
        
        # Random factor for diversity
        score += random.uniform(-2, 2)
        
        return max(0, score)

def create_book_with_eyes(winking=False):
    """Create an animated book character"""
    img = Image.new('RGB', (200, 250), color='#8B4513')
    draw = ImageDraw.Draw(img)
    
    # Book details
    draw.rectangle([10, 10, 190, 240], fill='#6B3410', outline='#4A2408', width=3)
    draw.rectangle([20, 20, 180, 50], fill='gold', outline='#B8860B', width=2)
    
    # Title on book
    draw.text((100, 35), "REAL-TIME", fill='#4A2408', anchor='mm')
    
    # Eyes
    if winking:
        # Left eye (winking)
        draw.arc([60, 80, 90, 110], start=0, end=180, fill='black', width=4)
        # Right eye (open)
        draw.ellipse([110, 80, 140, 110], fill='white', outline='black', width=2)
        draw.ellipse([118, 88, 132, 102], fill='blue')
        draw.ellipse([120, 90, 126, 96], fill='white')
    else:
        # Both eyes open
        for x_offset in [60, 110]:
            draw.ellipse([x_offset, 80, x_offset+30, 110], fill='white', outline='black', width=2)
            draw.ellipse([x_offset+8, 88, x_offset+22, 102], fill='blue')
            draw.ellipse([x_offset+10, 90, x_offset+16, 96], fill='white')
    
    # Mouth
    if winking:
        draw.arc([80, 130, 120, 150], start=0, end=180, fill='red', width=3)
    else:
        draw.ellipse([90, 135, 110, 145], fill='red')
    
    # Decorative elements
    draw.rectangle([30, 180, 170, 200], fill='#CD853F', outline='#8B4513', width=2)
    draw.text((100, 190), "DYNAMIC AI", fill='white', anchor='mm')
    
    return img

def display_advanced_book_card(book: Dict, rank: int, relevance_score: float = None):
    """Display enhanced book card with dynamic search results"""
    
    st.markdown(f'<div class="book-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        # Book cover with error handling
        try:
            if book['cover_url']:
                st.image(book['cover_url'], width=120)
            else:
                placeholder = create_placeholder_cover(book['title'], book['author'])
                st.image(placeholder, width=120)
        except:
            st.write("📚 Cover unavailable")
    
    with col2:
        # Book details
        st.markdown(f"### #{rank} {book['title']}")
        st.write(f"**👤 Author:** {book['author']}")
        st.write(f"**🏷️ Publisher:** {book.get('publisher', 'Unknown')}")
        
        # Advanced metadata
        col2a, col2b = st.columns(2)
        with col2a:
            st.write(f"**📂 Genre:** {book['genre']}")
            st.write(f"**⭐ Rating:** {book['rating']}/5.0")
        with col2b:
            st.write(f"**📅 Year:** {book['year']}")
            st.write(f"**📄 Pages:** {book.get('page_count', 'N/A')}")
        
        # Description
        st.write(f"**📖 Description:** {book['description']}")
        
        # Emotion tags
        if book['emotions']:
            emotions_html = " ".join([f"<span class='emotion-tag'>{emotion}</span>" for emotion in book['emotions']])
            st.markdown(f"**🎭 Reading Experience:** {emotions_html}", unsafe_allow_html=True)
        
        # Keywords
        if book.get('keywords'):
            st.write(f"**🔍 Key Topics:** {book['keywords']}")
    
    with col3:
        # Relevance score
        if relevance_score:
            st.markdown(f'<div class="score-badge">Relevance: {relevance_score:.1f}%</div>', unsafe_allow_html=True)
        
        # Quality indicators
        quality_score = book.get('quality_score', 0)
        st.markdown(f"**📊 Quality Score:** {quality_score:.1f}/6.0")
        
        # Purchase links
        st.markdown("**🛒 Get This Book:**")
        links = get_enhanced_purchase_links(book['title'], book['author'])
        
        for platform, url in list(links.items())[:4]:  # Show top 4 links
            st.markdown(f'<a href="{url}" target="_blank" class="link-button">{platform}</a>', unsafe_allow_html=True)
        
        # Show more links in expander
        with st.expander("More sources..."):
            for platform, url in list(links.items())[4:]:
                st.markdown(f'<a href="{url}" target="_blank" class="link-button">{platform}</a>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

def create_placeholder_cover(title: str, author: str):
    """Create a placeholder book cover"""
    img = Image.new('RGB', (120, 180), color=f'#{random.randint(100, 255):02x}{random.randint(100, 255):02x}{random.randint(100, 255):02x}')
    draw = ImageDraw.Draw(img)
    
    # Title (truncated)
    title_short = title[:20] + "..." if len(title) > 20 else title
    draw.text((60, 60), title_short, fill='white', anchor='mm')
    draw.text((60, 120), author, fill='white', anchor='mm')
    
    return img

def get_enhanced_purchase_links(title: str, author: str) -> Dict[str, str]:
    """Generate comprehensive purchase and free reading links"""
    title_encoded = requests.utils.quote(title)
    author_encoded = requests.utils.quote(author)
    search_query = f"{title_encoded}+{author_encoded}"
    
    return {
        "📚 Amazon": f"https://www.amazon.com/s?k={search_query}&i=stripbooks",
        "🏪 Barnes & Noble": f"https://www.barnesandnoble.com/s/{search_query}",
        "📖 Google Books": f"https://books.google.com/books?q={search_query}",
        "🛒 eBay": f"https://www.ebay.com/sch/i.html?_nkw={search_query}+book",
        "🏬 Walmart": f"https://www.walmart.com/search?q={search_query}+book",
        "📚 Open Library": f"https://openlibrary.org/search?q={title_encoded}",
        "📄 Project Gutenberg": f"https://www.gutenberg.org/ebooks/search/?query={search_query}",
        "🏛️ Internet Archive": f"https://archive.org/search.php?query={search_query}+book"
    }

def main():
    """Main application with dynamic search capabilities"""
    
    # Header with dynamic book character
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if 'winking' not in st.session_state:
            st.session_state.winking = False
        
        book_character = create_book_with_eyes(st.session_state.winking)
        st.image(book_character, caption="Real-Time Search Assistant!", width=200)
    
    with col2:
        st.markdown('<h1 class="main-header">📚 BookFinder AI - Dynamic Search</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time Google Books search powered by advanced AI!</p>', unsafe_allow_html=True)
    
    with col3:
        if st.button("👁️ Make me wink!", key="wink_button"):
            st.session_state.winking = not st.session_state.winking
            st.rerun()
    
    # Sidebar filters
    st.sidebar.header("🎯 Search Filters")
    
    # Dynamic search settings
    st.sidebar.markdown("### 🔍 Search Settings")
    search_intensity = st.sidebar.select_slider(
        "Search Intensity",
        options=["Light", "Moderate", "Intensive", "Deep"],
        value="Moderate",
        help="Higher intensity searches more sources but takes longer"
    )
    
    # Traditional filters
    categories = ['All', 'Fiction', 'Non-Fiction', 'Science', 'Technology', 'Business', 
                 'Self-Help', 'Biography', 'History', 'Philosophy', 'Psychology']
    selected_category = st.sidebar.selectbox("📂 Category Filter", categories)
    
    genres = ['Data Science', 'Programming', 'Business', 'Self-Help', 'Psychology', 
             'Science Fiction', 'Mystery/Thriller', 'Romance', 'Fantasy', 'Biography']
    selected_genres = st.sidebar.multiselect("🎭 Genre Filter", genres)
    
    min_rating = st.sidebar.slider("⭐ Minimum Rating", 1.0, 5.0, 3.0, 0.1)
    
    year_range = st.sidebar.slider("📅 Publication Year", 1980, 2024, (2000, 2024))
    
    page_range = st.sidebar.slider("📄 Page Count", 0, 1000, (50, 500))
    
    language = st.sidebar.selectbox("🌍 Language", ["All", "en", "es", "fr", "de", "it"])
    
    # Advanced search options
    st.sidebar.markdown("### 🔧 Advanced Options")
    include_older_books = st.sidebar.checkbox("Include books before 2000", value=False)
    prioritize_recent = st.sidebar.checkbox("Prioritize recent publications", value=True)
    avoid_duplicates = st.sidebar.checkbox("Avoid duplicate authors", value=True)
    
    # Main search interface
    st.header("🔮 Dynamic Book Search & Recommendations")
    
    # Information about dynamic search
    with st.expander("ℹ️ How Dynamic Search Works"):
        st.markdown("""
        **🚀 Real-Time Search Process:**
        1. **Intelligent Query Generation** - AI analyzes your input and generates multiple strategic search terms
        2. **Parallel Google Books API Calls** - Searches multiple queries simultaneously for speed
        3. **Smart Filtering** - Applies quality filters and user preferences
        4. **Advanced Scoring** - Ranks books based on relevance, quality, and user preferences
        5. **Real-Time Results** - No pre-loaded database - fresh results every time!
        
        **💡 Tips for Better Results:**
        - Be specific about what you want to learn or experience
        - Mention your current mood or goal
        - Include technical terms for specialized subjects
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        recent_books = st.text_area(
            "📚 Books you've read recently (comma-separated)",
            placeholder="e.g., Atomic Habits, Python Crash Course, The Lean Startup...",
            height=100,
            help="Helps avoid recommending books you've already read"
        )
        
    with col2:
        mood_input = st.text_area(
            "🎯 What do you want to read about? Be specific!",
            placeholder="e.g., I want to learn data science with Python, or I need practical business advice for startups, or I'm looking for inspiring biographies of tech leaders...",
            height=100,
            help="The more specific you are, the better the AI can find relevant books"
        )
    
    # Search intensity configuration
    search_configs = {
        "Light": {"queries": 3, "books_per_query": 10, "parallel_workers": 2},
        "Moderate": {"queries": 5, "books_per_query": 15, "parallel_workers": 3},
        "Intensive": {"queries": 8, "books_per_query": 20, "parallel_workers": 4},
        "Deep": {"queries": 12, "books_per_query": 25, "parallel_workers": 5}
    }
    
    # Get dynamic recommendations
    if st.button("🚀 Search Google Books in Real-Time!", type="primary", use_container_width=True):
        if not mood_input.strip():
            st.warning("⚠️ Please describe what kind of books you're looking for!")
            return
        
        # Prepare filters
        filters = {
            'category': selected_category,
            'genres': selected_genres,
            'min_rating': min_rating,
            'year_range': year_range,
            'page_range': page_range,
            'language': language if language != 'All' else None
        }
        
        # Process recent books
        recent_books_list = [book.strip() for book in recent_books.split(',') if book.strip()] if recent_books else []
        
        # Initialize dynamic recommendation engine
        rec_engine = DynamicRecommendationEngine()
        
        # Show search configuration
        config = search_configs[search_intensity]
        st.info(f"🔍 **Search Configuration:** {search_intensity} mode - {config['queries']} queries, {config['books_per_query']} books per query, {config['parallel_workers']} parallel workers")
        
        # Get recommendations
        try:
            recommendations = rec_engine.get_dynamic_recommendations(
                mood_input, 
                recent_books_list, 
                filters, 
                num_recommendations=5
            )
            
            if recommendations:
                st.header("🏆 Your Real-Time Book Recommendations")
                
                # Display search analytics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📚 Books Found", len(recommendations))
                with col2:
                    avg_rating = np.mean([book['rating'] for book in recommendations])
                    st.metric("⭐ Avg Rating", f"{avg_rating:.1f}/5.0")
                with col3:
                    avg_year = np.mean([book['year'] for book in recommendations])
                    st.metric("📅 Avg Year", f"{int(avg_year)}")
                
                # Display recommendations
                for i, book in enumerate(recommendations, 1):
                    # Calculate relevance score for display
                    relevance = rec_engine.calculate_relevance_score(
                        book, 
                        rec_engine.analyze_user_text(mood_input), 
                        recent_books_list
                    )
                    display_advanced_book_card(book, i, relevance)
                
                # Show search insights
                with st.expander("📊 Search Insights"):
                    genres_found = [book['genre'] for book in recommendations]
                    publishers_found = [book.get('publisher', 'Unknown') for book in recommendations]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Genres Found:**")
                        for genre in set(genres_found):
                            count = genres_found.count(genre)
                            st.write(f"• {genre}: {count} book(s)")
                    
                    with col2:
                        st.write("**Publishers:**")
                        for publisher in set(publishers_found)[:5]:
                            if publisher != 'Unknown':
                                st.write(f"• {publisher}")
            else:
                st.error("❌ No books found matching your criteria. Try:")
                st.markdown("""
                - Using different keywords
                - Broadening your search terms
                - Adjusting the filters in the sidebar
                - Trying a different search intensity
                """)
                
        except Exception as e:
            st.error(f"🚨 Search failed: {str(e)}")
            st.markdown("**Troubleshooting:**")
            st.markdown("- Check your internet connection")
            st.markdown("- Try simpler search terms")
            st.markdown("- Refresh the page and try again")
    
    # Additional features
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("🎲 Quick Searches")
        quick_searches = ["data science", "startup business", "self improvement", "science fiction", "biography"]
        selected_quick = st.selectbox("Choose a quick search:", ["Select..."] + quick_searches)
        
        if st.button("🚀 Quick Search", use_container_width=True) and selected_quick != "Select...":
            st.session_state.quick_search = selected_quick
            st.rerun()
    
    with col2:
        st.header("📈 Trending Topics")
        if st.button("🔥 Search AI & Technology", use_container_width=True):
            st.session_state.trending_search = "artificial intelligence machine learning technology"
            st.rerun()
        if st.button("💼 Search Business & Leadership", use_container_width=True):
            st.session_state.trending_search = "business leadership entrepreneurship"
            st.rerun()
    
    with col3:
        st.header("🔄 Search Tools")
        if st.button("🧹 Clear Search Cache", use_container_width=True):
            try:
                # Clear the search cache
                conn = sqlite3.connect("dynamic_books_cache.db")
                cursor = conn.cursor()
                cursor.execute("DELETE FROM search_cache")
                conn.commit()
                conn.close()
                st.success("✅ Search cache cleared!")
            except:
                st.warning("⚠️ Cache already empty or not found")
        
        if st.button("📊 Show Cache Stats", use_container_width=True):
            try:
                conn = sqlite3.connect("dynamic_books_cache.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM search_cache")
                count = cursor.fetchone()[0]
                conn.close()
                st.info(f"💾 {count} searches cached")
            except:
                st.info("💾 No cache data available")
    
    # Handle quick searches
    if hasattr(st.session_state, 'quick_search'):
        st.header(f"🚀 Quick Search Results for: {st.session_state.quick_search}")
        rec_engine = DynamicRecommendationEngine()
        quick_results = rec_engine.get_dynamic_recommendations(
            st.session_state.quick_search, [], {}, num_recommendations=3
        )
        for i, book in enumerate(quick_results, 1):
            display_advanced_book_card(book, i)
        del st.session_state.quick_search
    
    # Handle trending searches
    if hasattr(st.session_state, 'trending_search'):
        st.header(f"🔥 Trending Search Results")
        rec_engine = DynamicRecommendationEngine()
        trending_results = rec_engine.get_dynamic_recommendations(
            st.session_state.trending_search, [], {}, num_recommendations=3
        )
        for i, book in enumerate(trending_results, 1):
            display_advanced_book_card(book, i)
        del st.session_state.trending_search
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>📚 <strong>Real-Time Book Discovery with Dynamic Google Books Search</strong></p>
        <p>🚀 Live API Integration • Parallel Searches • Smart Filtering • Quality Scoring</p>
        <p><em>💡 Every search is fresh from Google Books - no stale data!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()