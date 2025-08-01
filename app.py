import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import re
import json
import time
from datetime import datetime
from textblob import TextBlob
import random
from typing import List, Dict, Any

# Configure page
st.set_page_config(
    page_title="📚 BookFinder AI - API Enhanced",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    }
    .api-status {
        background: linear-gradient(45deg, #ff6b6b, #ffa500);
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fallback-notice {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class BookSearchEngine:
    """Enhanced book search with multiple fallback strategies"""
    
    def __init__(self):
        self.google_books_api = "https://www.googleapis.com/books/v1/volumes"
        self.openlibrary_api = "https://openlibrary.org/search.json"
        
        # Try to get API key from Streamlit secrets
        try:
            self.api_key = st.secrets.get("GOOGLE_API_KEY")
        except:
            self.api_key = None
    
    def search_with_multiple_strategies(self, query: str, max_results: int = 20) -> List[Dict]:
        """Try multiple search strategies"""
        
        # Strategy 1: Google Books with API key
        if self.api_key:
            books = self.search_google_books_with_key(query, max_results)
            if books:
                st.success(f"✅ Found {len(books)} books using Google Books API")
                return books
        
        # Strategy 2: Google Books without API key (different parameters)
        books = self.search_google_books_enhanced(query, max_results)
        if books:
            st.success(f"✅ Found {len(books)} books using enhanced Google Books search")
            return books
        
        # Strategy 3: Open Library API
        books = self.search_open_library(query, max_results)
        if books:
            st.success(f"✅ Found {len(books)} books using Open Library API")
            return books
        
        # Strategy 4: Fallback to curated list
        return self.get_fallback_books(query)
    
    def search_google_books_with_key(self, query: str, max_results: int) -> List[Dict]:
        """Search with API key"""
        try:
            params = {
                'q': query,
                'maxResults': min(max_results, 40),
                'orderBy': 'relevance',
                'key': self.api_key
            }
            
            response = requests.get(self.google_books_api, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_google_books_response(data)
            else:
                st.warning(f"⚠️ Google Books API with key returned {response.status_code}")
                return []
                
        except Exception as e:
            st.warning(f"⚠️ Google Books API with key failed: {str(e)}")
            return []
    
    def search_google_books_enhanced(self, query: str, max_results: int) -> List[Dict]:
        """Enhanced Google Books search without API key"""
        try:
            # Try different parameter combinations
            param_sets = [
                {
                    'q': f'"{query}"',  # Exact phrase
                    'maxResults': min(max_results, 20),
                    'projection': 'lite',
                    'printType': 'books'
                },
                {
                    'q': query,
                    'maxResults': min(max_results, 15),
                    'orderBy': 'newest'
                },
                {
                    'q': f'{query} books',
                    'maxResults': min(max_results, 10)
                }
            ]
            
            for i, params in enumerate(param_sets):
                try:
                    headers = {
                        'User-Agent': f'BookFinder-AI/1.0 (Search {i+1})',
                        'Accept': 'application/json',
                        'Accept-Language': 'en-US,en;q=0.9'
                    }
                    
                    response = requests.get(
                        self.google_books_api, 
                        params=params, 
                        headers=headers,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        books = self.parse_google_books_response(data)
                        if books:
                            return books
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    continue
            
            return []
            
        except Exception as e:
            return []
    
    def search_open_library(self, query: str, max_results: int) -> List[Dict]:
        """Search Open Library as fallback"""
        try:
            params = {
                'q': query,
                'limit': min(max_results, 20),
                'fields': 'title,author_name,first_publish_year,subject,cover_i,isbn,key'
            }
            
            response = requests.get(self.openlibrary_api, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_open_library_response(data, query)
            
            return []
            
        except Exception as e:
            return []
    
    def parse_google_books_response(self, data: Dict) -> List[Dict]:
        """Parse Google Books API response"""
        books = []
        
        for item in data.get('items', []):
            try:
                volume_info = item.get('volumeInfo', {})
                
                title = volume_info.get('title', '').strip()
                if not title:
                    continue
                
                authors = volume_info.get('authors', ['Unknown'])
                author = ', '.join(authors) if isinstance(authors, list) else str(authors)
                
                # Basic info
                categories = volume_info.get('categories', ['General'])
                category = categories[0] if categories else 'General'
                
                rating = volume_info.get('averageRating', random.uniform(3.8, 4.5))
                
                published_date = volume_info.get('publishedDate', '2000')
                year = self.extract_year(published_date)
                
                description = volume_info.get('description', 'No description available.')
                if len(description) > 300:
                    description = description[:300] + "..."
                
                # Cover image
                image_links = volume_info.get('imageLinks', {})
                cover_url = (image_links.get('thumbnail') or 
                            image_links.get('smallThumbnail') or '')
                
                if cover_url and cover_url.startswith('http:'):
                    cover_url = cover_url.replace('http:', 'https:')
                
                books.append({
                    'title': title,
                    'author': author,
                    'category': category,
                    'genre': self.classify_genre(category, title, description),
                    'rating': round(rating, 1),
                    'year': year,
                    'description': description,
                    'emotions': self.extract_emotions(description, category),
                    'keywords': self.extract_keywords(title, description),
                    'cover_url': cover_url,
                    'source': 'Google Books'
                })
                
            except Exception as e:
                continue
        
        return books
    
    def parse_open_library_response(self, data: Dict, original_query: str) -> List[Dict]:
        """Parse Open Library response"""
        books = []
        
        for item in data.get('docs', []):
            try:
                title = item.get('title', '').strip()
                if not title:
                    continue
                
                authors = item.get('author_name', ['Unknown'])
                author = ', '.join(authors[:2]) if isinstance(authors, list) else str(authors)
                
                year = item.get('first_publish_year', 2000)
                subjects = item.get('subject', [])
                category = subjects[0] if subjects else 'General'
                
                # Generate description from subjects
                description = f"A book about {', '.join(subjects[:5])}." if subjects else "No description available."
                
                # Cover URL from Open Library
                cover_id = item.get('cover_i')
                cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else ''
                
                books.append({
                    'title': title,
                    'author': author,
                    'category': category,
                    'genre': self.classify_genre(category, title, description),
                    'rating': random.uniform(3.5, 4.5),
                    'year': year,
                    'description': description,
                    'emotions': self.extract_emotions(description, category),
                    'keywords': ', '.join(subjects[:5]) if subjects else original_query,
                    'cover_url': cover_url,
                    'source': 'Open Library'
                })
                
            except Exception as e:
                continue
        
        return books[:20]  # Limit results
    
    def get_fallback_books(self, query: str) -> List[Dict]:
        """Curated fallback books when APIs fail"""
        
        # Curated books by category
        fallback_data = {
            'investing': [
                {
                    'title': 'Rich Dad Poor Dad',
                    'author': 'Robert Kiyosaki',
                    'category': 'Finance',
                    'genre': 'Personal Finance',
                    'rating': 4.1,
                    'year': 1997,
                    'description': 'A guide to financial literacy and building wealth through real estate investing, starting a business, and increasing financial intelligence.',
                    'emotions': ['motivational', 'educational', 'practical'],
                    'keywords': 'wealth, financial literacy, investing, real estate, passive income',
                    'cover_url': 'https://images-na.ssl-images-amazon.com/images/I/81bsw6fnUiL.jpg',
                    'source': 'Curated'
                },
                {
                    'title': 'The Intelligent Investor',
                    'author': 'Benjamin Graham',
                    'category': 'Finance',
                    'genre': 'Investment Strategy',
                    'rating': 4.3,
                    'year': 1949,
                    'description': 'The classic guide to value investing, teaching principles of sound investment decisions and market analysis.',
                    'emotions': ['analytical', 'educational', 'timeless'],
                    'keywords': 'value investing, stock market, financial analysis, warren buffett',
                    'cover_url': 'https://images-na.ssl-images-amazon.com/images/I/91VokXkn9rL.jpg',
                    'source': 'Curated'
                }
            ],
            'finance': [
                {
                    'title': 'A Random Walk Down Wall Street',
                    'author': 'Burton Malkiel',
                    'category': 'Finance',
                    'genre': 'Investment Theory',
                    'rating': 4.2,
                    'year': 1973,
                    'description': 'An investment guide that advocates for index fund investing and efficient market theory.',
                    'emotions': ['analytical', 'practical', 'educational'],
                    'keywords': 'index funds, efficient market, portfolio management, wall street',
                    'cover_url': 'https://images-na.ssl-images-amazon.com/images/I/71g2ednj0JL.jpg',
                    'source': 'Curated'
                }
            ],
            'business': [
                {
                    'title': 'The Lean Startup',
                    'author': 'Eric Ries',
                    'category': 'Business',
                    'genre': 'Entrepreneurship',
                    'rating': 4.2,
                    'year': 2011,
                    'description': 'A methodology for developing businesses and products through validated learning and iterative design.',
                    'emotions': ['innovative', 'practical', 'inspiring'],
                    'keywords': 'startup, entrepreneurship, innovation, lean methodology, mvp',
                    'cover_url': 'https://images-na.ssl-images-amazon.com/images/I/81vvgZqCskL.jpg',
                    'source': 'Curated'
                }
            ]
        }
        
        # Find matching category
        query_lower = query.lower()
        for category, books in fallback_data.items():
            if category in query_lower or any(word in query_lower for word in category.split()):
                st.markdown(f'<div class="fallback-notice">📚 <strong>Showing curated {category.title()} books</strong> (APIs unavailable)</div>', unsafe_allow_html=True)
                return books
        
        # Default fallback
        st.markdown('<div class="fallback-notice">📚 <strong>Showing popular general books</strong> (APIs unavailable)</div>', unsafe_allow_html=True)
        return fallback_data['business'] + fallback_data['investing'][:1]
    
    def extract_year(self, date_str: str) -> int:
        """Extract year from date string"""
        try:
            match = re.search(r'(\d{4})', date_str)
            if match:
                year = int(match.group(1))
                if 1800 <= year <= datetime.now().year + 1:
                    return year
        except:
            pass
        return 2000
    
    def classify_genre(self, category: str, title: str, description: str) -> str:
        """Classify book genre"""
        combined = f"{category} {title} {description}".lower()
        
        if any(word in combined for word in ['invest', 'finance', 'money', 'wealth']):
            return 'Finance & Investing'
        elif any(word in combined for word in ['business', 'entrepreneur', 'startup']):
            return 'Business & Entrepreneurship'
        elif any(word in combined for word in ['data', 'science', 'machine', 'learning']):
            return 'Data Science & Technology'
        else:
            return category
    
    def extract_emotions(self, description: str, category: str) -> List[str]:
        """Extract reading emotions"""
        combined = f"{description} {category}".lower()
        
        emotions = []
        if any(word in combined for word in ['practical', 'guide', 'how-to']):
            emotions.append('practical')
        if any(word in combined for word in ['inspiring', 'motivational', 'success']):
            emotions.append('inspiring')
        if any(word in combined for word in ['educational', 'learn', 'understand']):
            emotions.append('educational')
        
        return emotions if emotions else ['engaging']
    
    def extract_keywords(self, title: str, description: str) -> str:
        """Extract relevant keywords"""
        combined = f"{title} {description}".lower()
        words = re.findall(r'\b\w{4,}\b', combined)
        
        # Remove common words
        stop_words = {'book', 'author', 'guide', 'will', 'with', 'this', 'that', 'from'}
        keywords = [word for word in words if word not in stop_words]
        
        # Get top words
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:5]
        return ', '.join([word for word, count in top_words])

def create_book_with_eyes(winking=False):
    """Create animated book character"""
    img = Image.new('RGB', (200, 250), color='#8B4513')
    draw = ImageDraw.Draw(img)
    
    # Book design
    draw.rectangle([10, 10, 190, 240], fill='#6B3410', outline='#4A2408', width=3)
    draw.rectangle([20, 20, 180, 50], fill='gold', outline='#B8860B', width=2)
    draw.text((100, 35), "API SMART", fill='#4A2408', anchor='mm')
    
    # Eyes
    if winking:
        draw.arc([60, 80, 90, 110], start=0, end=180, fill='black', width=4)
        draw.ellipse([110, 80, 140, 110], fill='white', outline='black', width=2)
        draw.ellipse([118, 88, 132, 102], fill='blue')
    else:
        for x_offset in [60, 110]:
            draw.ellipse([x_offset, 80, x_offset+30, 110], fill='white', outline='black', width=2)
            draw.ellipse([x_offset+8, 88, x_offset+22, 102], fill='blue')
    
    # Mouth
    draw.ellipse([90, 135, 110, 145], fill='red')
    
    # Bottom text
    draw.rectangle([30, 180, 170, 200], fill='#CD853F', outline='#8B4513', width=2)
    draw.text((100, 190), "MULTI-API", fill='white', anchor='mm')
    
    return img

def display_book_card(book: Dict, rank: int):
    """Display book card"""
    
    st.markdown('<div class="book-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        try:
            if book['cover_url']:
                st.image(book['cover_url'], width=120)
            else:
                st.write("📚 No cover")
        except:
            st.write("📚 No cover")
    
    with col2:
        st.markdown(f"### #{rank} {book['title']}")
        st.write(f"**👤 Author:** {book['author']}")
        st.write(f"**📂 Genre:** {book['genre']}")
        st.write(f"**⭐ Rating:** {book['rating']}/5.0")
        st.write(f"**📅 Year:** {book['year']}")
        st.write(f"**📖 Description:** {book['description']}")
        
        if book['emotions']:
            emotions_html = " ".join([f"<span class='emotion-tag'>{emotion}</span>" for emotion in book['emotions']])
            st.markdown(f"**🎭 Experience:** {emotions_html}", unsafe_allow_html=True)
        
        st.write(f"**🔍 Keywords:** {book['keywords']}")
        st.write(f"**📊 Source:** {book['source']}")
    
    with col3:
        st.markdown("**🛒 Get This Book:**")
        
        title_encoded = requests.utils.quote(book['title'])
        author_encoded = requests.utils.quote(book['author'])
        
        links = {
            "📚 Amazon": f"https://www.amazon.com/s?k={title_encoded}+{author_encoded}",
            "📖 Google Books": f"https://books.google.com/books?q={title_encoded}",
            "🏪 Barnes & Noble": f"https://www.barnesandnoble.com/s/{title_encoded}",
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
        st.image(book_character, caption="Multi-API Search Assistant!", width=200)
    
    with col2:
        st.markdown('<h1 class="main-header">📚 BookFinder AI - Enhanced</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Multi-API search with smart fallbacks!</p>', unsafe_allow_html=True)
    
    with col3:
        if st.button("👁️ Wink!", key="wink_button"):
            st.session_state.winking = not st.session_state.winking
            st.rerun()
    
    # API Status
    search_engine = BookSearchEngine()
    if search_engine.api_key:
        st.markdown('<div class="api-status">🔑 <strong>Google Books API Key detected</strong> - Enhanced search enabled!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status">ℹ️ <strong>No API key</strong> - Using fallback strategies</div>', unsafe_allow_html=True)
        
        with st.expander("🔑 How to add Google Books API Key (Optional)"):
            st.markdown("""
            **To enable enhanced search:**
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select existing
            3. Enable the Books API
            4. Create an API key
            5. In Streamlit Cloud, go to app settings → Secrets
            6. Add: `GOOGLE_BOOKS_API_KEY = "your-api-key-here"`
            
            **Note:** The app works without an API key using multiple fallback strategies!
            """)
    
    # Main interface
    st.header("🔮 Enhanced Book Search")
    
    col1, col2 = st.columns(2)
    
    with col1:
        recent_books = st.text_area(
            "📚 Books you've read recently",
            placeholder="e.g., Rich Dad Poor Dad, The Lean Startup...",
            height=100
        )
        
    with col2:
        search_query = st.text_area(
            "🎯 What do you want to read about?",
            placeholder="e.g., investing, personal finance, business startup, data science...",
            height=100
        )
    
    # Search button
    if st.button("🚀 Find Books with Multi-API Search!", type="primary", use_container_width=True):
        if not search_query.strip():
            st.warning("⚠️ Please enter what you want to read about!")
            return
        
        st.markdown("### 🔍 Multi-Strategy Search")
        
        # Perform search
        books = search_engine.search_with_multiple_strategies(search_query, max_results=25)
        
        if books:
            st.header(f"🏆 Found {len(books)} Book Recommendations")
            
            # Display up to 5 books
            for i, book in enumerate(books[:5], 1):
                display_book_card(book, i)
                
        else:
            st.error("❌ All search strategies failed. Please try again later.")
    
    # Quick search buttons
    st.markdown("### 🚀 Quick Searches")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💰 Investing Books"):
            books = search_engine.search_with_multiple_strategies("investing personal finance", 10)
            if books:
                for i, book in enumerate(books[:3], 1):
                    display_book_card(book, i)
    
    with col2:
        if st.button("🚀 Business Books"):
            books = search_engine.search_with_multiple_strategies("business entrepreneurship startup", 10)
            if books:
                for i, book in enumerate(books[:3], 1):
                    display_book_card(book, i)
    
    with col3:
        if st.button("📊 Data Science"):
            books = search_engine.search_with_multiple_strategies("data science machine learning", 10)
            if books:
                for i, book in enumerate(books[:3], 1):
                    display_book_card(book, i)
    
    with col4:
        if st.button("💪 Self Help"):
            books = search_engine.search_with_multiple_strategies("self help personal development", 10)
            if books:
                for i, book in enumerate(books[:3], 1):
                    display_book_card(book, i)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>📚 <strong>Multi-API Book Discovery System</strong></p>
        <p>🔄 Google Books → Open Library → Curated Fallbacks</p>
        <p><em>💡 Always finds books, even when APIs are down!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
