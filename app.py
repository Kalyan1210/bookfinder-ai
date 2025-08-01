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

class StreamlitCloudBookSearchEngine:
    """Streamlit Cloud compatible book search engine"""
    
    def __init__(self):
        self.google_books_api = "https://www.googleapis.com/books/v1/volumes"
        self.rate_limit_delay = 0.5  # Increased delay for Streamlit Cloud
        
    def generate_search_queries(self, user_input: str, recent_books: List[str]) -> List[str]:
        """Generate intelligent search queries based on user input"""
        
        # Subject mapping for intelligent query generation
        subject_patterns = {
            'data_science': ['data science', 'machine learning', 'artificial intelligence', 'python programming'],
            'programming': ['programming', 'coding', 'software development', 'computer science'],
            'business': ['business', 'entrepreneurship', 'startup', 'management'],
            'investing': ['investing', 'finance', 'personal finance', 'stock market'],
            'psychology': ['psychology', 'mental health', 'cognitive science', 'self help'],
            'history': ['history', 'biography', 'world history', 'historical'],
            'science': ['physics', 'chemistry', 'biology', 'science'],
            'philosophy': ['philosophy', 'ethics', 'meaning', 'wisdom'],
            'fiction': ['fiction', 'novel', 'story', 'literature'],
            'self_help': ['self help', 'personal development', 'productivity', 'motivation'],
        }
        
        # Generate base queries from detected subjects
        queries = []
        user_text = user_input.lower()
        
        # Direct keyword extraction
        for subject, keywords in subject_patterns.items():
            for keyword in keywords:
                if keyword in user_text:
                    queries.extend(keywords[:2])  # Take top 2 related terms
                    break
        
        # Add direct user words if no patterns matched
        if not queries:
            words = user_text.split()
            meaningful_words = [word for word in words if len(word) > 3]
            queries.extend(meaningful_words[:3])
        
        # Add some general high-quality terms
        if not queries:
            queries = ['bestseller', 'popular books', 'highly rated']
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(queries))[:5]
        
        return unique_queries
    
    def search_google_books_sequential(self, query: str, max_results: int = 20) -> List[Dict]:
        """Sequential search for Streamlit Cloud compatibility"""
        
        try:
            # Build search URL with better parameters
            search_params = {
                'q': query,
                'maxResults': min(max_results, 40),  # Google Books API limit
                'orderBy': 'relevance',
                'printType': 'books',
                'projection': 'lite'  # Faster response
            }
            
            # Add progress indicator
            st.write(f"🔍 Searching for: **{query}**...")
            
            response = requests.get(
                self.google_books_api, 
                params=search_params, 
                timeout=15,  # Increased timeout
                headers={'User-Agent': 'BookFinder-AI/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                books = []
                
                items = data.get('items', [])
                st.write(f"✅ Found {len(items)} results for '{query}'")
                
                for item in items:
                    book_info = self.parse_google_book_simple(item)
                    if book_info:
                        books.append(book_info)
                
                # Rate limiting for Streamlit Cloud
                time.sleep(self.rate_limit_delay)
                
                return books
            else:
                st.warning(f"⚠️ API returned status {response.status_code} for query: {query}")
                return []
                
        except requests.exceptions.Timeout:
            st.error(f"⏰ Timeout searching for: {query}")
            return []
        except Exception as e:
            st.error(f"❌ Error searching '{query}': {str(e)}")
            return []
    
    def parse_google_book_simple(self, item: Dict) -> Dict:
        """Simplified parsing for better reliability"""
        try:
            volume_info = item.get('volumeInfo', {})
            
            # Basic info with fallbacks
            title = volume_info.get('title', '').strip()
            if not title:
                return None
                
            authors = volume_info.get('authors', ['Unknown Author'])
            author = ', '.join(authors) if isinstance(authors, list) else str(authors)
            
            # Categories
            categories = volume_info.get('categories', ['General'])
            category = categories[0] if categories else 'General'
            
            # Rating with fallback
            rating = volume_info.get('averageRating')
            if rating is None:
                rating = random.uniform(3.8, 4.5)  # Reasonable fallback
            
            # Year extraction
            published_date = volume_info.get('publishedDate', '2000')
            year = self.extract_year_simple(published_date)
            
            # Description
            description = volume_info.get('description', 'No description available.')
            if len(description) > 300:
                description = description[:300] + "..."
            
            # Simple emotion extraction
            emotions = self.extract_emotions_simple(description, category)
            
            # Cover image
            image_links = volume_info.get('imageLinks', {})
            cover_url = (image_links.get('thumbnail') or 
                        image_links.get('smallThumbnail') or '')
            
            if cover_url and cover_url.startswith('http:'):
                cover_url = cover_url.replace('http:', 'https:')
            
            return {
                'title': title,
                'author': author,
                'category': category,
                'genre': self.simple_genre_classification(category, title),
                'rating': round(rating, 1),
                'year': year,
                'description': description,
                'emotions': emotions,
                'keywords': self.extract_simple_keywords(title, description),
                'cover_url': cover_url,
                'google_id': item.get('id', ''),
                'page_count': volume_info.get('pageCount', 0),
                'quality_score': self.simple_quality_score(volume_info)
            }
            
        except Exception as e:
            return None
    
    def extract_year_simple(self, published_date: str) -> int:
        """Simple year extraction"""
        try:
            year_match = re.search(r'(\d{4})', published_date)
            if year_match:
                year = int(year_match.group(1))
                if 1800 <= year <= datetime.now().year + 1:
                    return year
        except:
            pass
        return 2000
    
    def extract_emotions_simple(self, description: str, category: str) -> List[str]:
        """Simple emotion extraction"""
        combined_text = f"{description} {category}".lower()
        
        emotion_patterns = {
            'educational': ['learn', 'guide', 'tutorial', 'education', 'study'],
            'inspiring': ['inspiring', 'motivational', 'success', 'achievement'],
            'practical': ['practical', 'hands-on', 'how-to', 'step-by-step'],
            'entertaining': ['entertaining', 'funny', 'story', 'adventure'],
            'thought-provoking': ['thought', 'philosophy', 'deep', 'analysis'],
            'professional': ['business', 'career', 'professional', 'work']
        }
        
        detected = []
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                detected.append(emotion)
        
        return detected[:3] if detected else ['engaging']
    
    def simple_genre_classification(self, category: str, title: str) -> str:
        """Simple genre classification"""
        combined = f"{category} {title}".lower()
        
        if any(word in combined for word in ['data', 'science', 'machine', 'learning', 'python']):
            return 'Data Science'
        elif any(word in combined for word in ['business', 'entrepreneur', 'startup']):
            return 'Business'
        elif any(word in combined for word in ['invest', 'finance', 'money', 'stock']):
            return 'Finance'
        elif any(word in combined for word in ['program', 'coding', 'software']):
            return 'Programming'
        elif any(word in combined for word in ['self', 'help', 'personal', 'development']):
            return 'Self-Help'
        else:
            return category
    
    def extract_simple_keywords(self, title: str, description: str) -> str:
        """Simple keyword extraction"""
        combined = f"{title} {description}".lower()
        words = re.findall(r'\b\w{4,}\b', combined)  # Words with 4+ characters
        
        # Remove common words
        stop_words = {'book', 'author', 'story', 'will', 'with', 'this', 'that', 'from', 'they', 'have', 'been'}
        keywords = [word for word in words if word not in stop_words]
        
        # Get most frequent
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:6]
        return ', '.join([word for word, count in top_words])
    
    def simple_quality_score(self, volume_info: Dict) -> float:
        """Simple quality scoring"""
        score = 3.0  # Base score
        
        if volume_info.get('description'):
            score += 1.0
        if volume_info.get('averageRating'):
            score += 1.0
        if volume_info.get('imageLinks'):
            score += 0.5
        if volume_info.get('pageCount', 0) > 50:
            score += 0.5
        
        return score
    
    def search_multiple_queries_sequential(self, queries: List[str], books_per_query: int = 15) -> List[Dict]:
        """Search multiple queries sequentially for Streamlit Cloud"""
        all_books = []
        
        progress_bar = st.progress(0)
        
        for i, query in enumerate(queries):
            try:
                books = self.search_google_books_sequential(query, books_per_query)
                all_books.extend(books)
                
                progress_bar.progress((i + 1) / len(queries))
                
                # Show intermediate results
                st.write(f"📚 Collected {len(all_books)} books so far...")
                
            except Exception as e:
                st.warning(f"⚠️ Skipped query '{query}': {str(e)}")
                continue
        
        progress_bar.empty()
        
        # Remove duplicates
        seen = set()
        unique_books = []
        for book in all_books:
            identifier = f"{book['title']}_{book['author']}"
            if identifier not in seen:
                seen.add(identifier)
                unique_books.append(book)
        
        # Filter by quality
        quality_books = [book for book in unique_books if book.get('quality_score', 0) >= 3.0]
        
        return quality_books

class SimplifiedRecommendationEngine:
    """Simplified recommendation engine for Streamlit Cloud"""
    
    def __init__(self):
        self.search_engine = StreamlitCloudBookSearchEngine()
    
    def get_recommendations(self, user_input: str, recent_books: List[str], 
                          filters: Dict, num_recommendations: int = 5) -> List[Dict]:
        """Get recommendations with simplified processing"""
        
        # Generate search queries
        search_queries = self.search_engine.generate_search_queries(user_input, recent_books)
        
        # Display search strategy
        st.markdown("### 🔍 Dynamic Search Strategy")
        st.markdown('<div class="search-status">🚀 Performing real-time searches on Google Books API...</div>', unsafe_allow_html=True)
        
        search_display = " ".join([f'<span class="dynamic-search-indicator">{query}</span>' for query in search_queries])
        st.markdown(f"**Search Queries:** {search_display}", unsafe_allow_html=True)
        
        # Perform sequential searches (Streamlit Cloud compatible)
        books = self.search_engine.search_multiple_queries_sequential(search_queries, books_per_query=20)
        
        if not books:
            st.error("❌ No books found with current search terms. Try different keywords.")
            return []
        
        st.success(f"✅ Found {len(books)} books from real-time search!")
        
        # Apply filters
        filtered_books = self.apply_simple_filters(books, filters)
        
        # Simple scoring
        scored_books = self.score_books_simple(filtered_books, user_input, recent_books)
        
        return scored_books[:num_recommendations]
    
    def apply_simple_filters(self, books: List[Dict], filters: Dict) -> List[Dict]:
        """Apply user filters"""
        filtered = books.copy()
        
        if filters.get('category') and filters['category'] != 'All':
            filtered = [book for book in filtered if book['category'] == filters['category']]
        
        if filters.get('genres'):
            filtered = [book for book in filtered if book['genre'] in filters['genres']]
        
        if filters.get('min_rating'):
            filtered = [book for book in filtered if book['rating'] >= filters['min_rating']]
        
        if filters.get('year_range'):
            year_min, year_max = filters['year_range']
            filtered = [book for book in filtered if year_min <= book['year'] <= year_max]
        
        return filtered
    
    def score_books_simple(self, books: List[Dict], user_input: str, recent_books: List[str]) -> List[Dict]:
        """Simple book scoring"""
        
        scored_books = []
        user_words = set(user_input.lower().split())
        
        for book in books:
            score = 0.0
            
            # Base rating score
            score += book['rating'] * 10
            
            # Text relevance
            book_text = f"{book['title']} {book['description']} {book['keywords']}".lower()
            book_words = set(book_text.split())
            
            common_words = user_words & book_words
            if len(user_words) > 0:
                relevance = len(common_words) / len(user_words)
                score += relevance * 30
            
            # Quality bonus
            score += book.get('quality_score', 0) * 5
            
            # Avoid recently read
            if any(recent.lower() in book['title'].lower() for recent in recent_books):
                score -= 20
            
            # Random factor
            score += random.uniform(-2, 2)
            
            scored_books.append((book, score))
        
        # Sort by score
        scored_books.sort(key=lambda x: x[1], reverse=True)
        
        return [book for book, score in scored_books]

def create_book_with_eyes(winking=False):
    """Create animated book character"""
    img = Image.new('RGB', (200, 250), color='#8B4513')
    draw = ImageDraw.Draw(img)
    
    # Book details
    draw.rectangle([10, 10, 190, 240], fill='#6B3410', outline='#4A2408', width=3)
    draw.rectangle([20, 20, 180, 50], fill='gold', outline='#B8860B', width=2)
    
    # Title
    draw.text((100, 35), "REAL-TIME", fill='#4A2408', anchor='mm')
    
    # Eyes
    if winking:
        draw.arc([60, 80, 90, 110], start=0, end=180, fill='black', width=4)
        draw.ellipse([110, 80, 140, 110], fill='white', outline='black', width=2)
        draw.ellipse([118, 88, 132, 102], fill='blue')
        draw.ellipse([120, 90, 126, 96], fill='white')
    else:
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

def display_book_card(book: Dict, rank: int, relevance_score: float = None):
    """Display book card"""
    
    st.markdown(f'<div class="book-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        try:
            if book['cover_url']:
                st.image(book['cover_url'], width=120)
            else:
                st.write("📚 Cover unavailable")
        except:
            st.write("📚 Cover unavailable")
    
    with col2:
        st.markdown(f"### #{rank} {book['title']}")
        st.write(f"**👤 Author:** {book['author']}")
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.write(f"**📂 Genre:** {book['genre']}")
            st.write(f"**⭐ Rating:** {book['rating']}/5.0")
        with col2b:
            st.write(f"**📅 Year:** {book['year']}")
            st.write(f"**📄 Pages:** {book.get('page_count', 'N/A')}")
        
        st.write(f"**📖 Description:** {book['description']}")
        
        if book['emotions']:
            emotions_html = " ".join([f"<span class='emotion-tag'>{emotion}</span>" for emotion in book['emotions']])
            st.markdown(f"**🎭 Reading Experience:** {emotions_html}", unsafe_allow_html=True)
        
        if book.get('keywords'):
            st.write(f"**🔍 Key Topics:** {book['keywords']}")
    
    with col3:
        if relevance_score:
            st.markdown(f'<div class="score-badge">Match: {relevance_score:.1f}%</div>', unsafe_allow_html=True)
        
        st.markdown("**🛒 Get This Book:**")
        
        # Purchase links
        title_encoded = requests.utils.quote(book['title'])
        author_encoded = requests.utils.quote(book['author'])
        search_query = f"{title_encoded}+{author_encoded}"
        
        links = {
            "📚 Amazon": f"https://www.amazon.com/s?k={search_query}&i=stripbooks",
            "📖 Google Books": f"https://books.google.com/books?q={search_query}",
            "🏪 Barnes & Noble": f"https://www.barnesandnoble.com/s/{search_query}",
            "📚 Open Library": f"https://openlibrary.org/search?q={title_encoded}"
        }
        
        for platform, url in links.items():
            st.markdown(f'<a href="{url}" target="_blank" class="link-button">{platform}</a>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

def main():
    """Main application"""
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if 'winking' not in st.session_state:
            st.session_state.winking = False
        
        book_character = create_book_with_eyes(st.session_state.winking)
        st.image(book_character, caption="Real-Time Search Assistant!", width=200)
    
    with col2:
        st.markdown('<h1 class="main-header">📚 BookFinder AI - Dynamic Search</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time Google Books search powered by AI!</p>', unsafe_allow_html=True)
    
    with col3:
        if st.button("👁️ Make me wink!", key="wink_button"):
            st.session_state.winking = not st.session_state.winking
            st.rerun()
    
    # Sidebar filters
    st.sidebar.header("🎯 Search Filters")
    
    search_intensity = st.sidebar.select_slider(
        "Search Intensity",
        options=["Light", "Moderate", "Intensive"],
        value="Moderate"
    )
    
    categories = ['All', 'Fiction', 'Non-Fiction', 'Science', 'Technology', 'Business', 
                 'Self-Help', 'Biography', 'History', 'Philosophy']
    selected_category = st.sidebar.selectbox("📂 Category Filter", categories)
    
    genres = ['Data Science', 'Programming', 'Business', 'Finance', 'Self-Help', 
             'Science Fiction', 'Biography', 'History']
    selected_genres = st.sidebar.multiselect("🎭 Genre Filter", genres)
    
    min_rating = st.sidebar.slider("⭐ Minimum Rating", 1.0, 5.0, 3.0, 0.1)
    year_range = st.sidebar.slider("📅 Publication Year", 1980, 2024, (2000, 2024))
    
    # Main interface
    st.header("🔮 Dynamic Book Search & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        recent_books = st.text_area(
            "📚 Books you've read recently (comma-separated)",
            placeholder="e.g., Atomic Habits, Python Crash Course, Rich Dad Poor Dad...",
            height=100
        )
        
    with col2:
        mood_input = st.text_area(
            "🎯 What do you want to read about? Be specific!",
            placeholder="e.g., I want to learn about investing and personal finance, or I need practical business advice for entrepreneurs...",
            height=100
        )
    
    # Search button
    if st.button("🚀 Search Google Books in Real-Time!", type="primary", use_container_width=True):
        if not mood_input.strip():
            st.warning("⚠️ Please describe what kind of books you're looking for!")
            return
        
        # Prepare filters
        filters = {
            'category': selected_category,
            'genres': selected_genres,
            'min_rating': min_rating,
            'year_range': year_range
        }
        
        recent_books_list = [book.strip() for book in recent_books.split(',') if book.strip()] if recent_books else []
        
        # Get recommendations
        rec_engine = SimplifiedRecommendationEngine()
        
        try:
            recommendations = rec_engine.get_recommendations(
                mood_input, 
                recent_books_list, 
                filters, 
                num_recommendations=5
            )
            
            if recommendations:
                st.header("🏆 Your Real-Time Book Recommendations")
                
                for i, book in enumerate(recommendations, 1):
                    # Calculate simple relevance score
                    user_words = set(mood_input.lower().split())
                    book_text = f"{book['title']} {book['description']}".lower()
                    book_words = set(book_text.split())
                    relevance = len(user_words & book_words) / max(len(user_words), 1) * 100
                    
                    display_book_card(book, i, relevance)
            else:
                st.error("❌ No books found. Try:")
                st.markdown("- Using simpler keywords")
                st.markdown("- Broadening your search terms") 
                st.markdown("- Trying different topics")
                
        except Exception as e:
            st.error(f"🚨 Search failed: {str(e)}")
            st.markdown("**Try:**")
            st.markdown("- Simpler search terms")
            st.markdown("- Refresh the page")
            st.markdown("- Check your internet connection")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>📚 <strong>Real-Time Book Discovery with Google Books API</strong></p>
        <p>🚀 Sequential Processing • Smart Filtering • Quality Scoring</p>
        <p><em>💡 Streamlit Cloud optimized for reliable performance!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
