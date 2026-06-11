import pandas as pd
import os
import sys
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# Load Environment Variables
load_dotenv()
GOOGLE_BOOKS_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

def fetch_google_books_data(query):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {'q': query}
    if GOOGLE_BOOKS_KEY:
        params['key'] = GOOGLE_BOOKS_KEY
        
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get('totalItems', 0) > 0:
            return data['items'][0]['volumeInfo']
    except Exception as e:
        print(f"Error fetching from Google Books: {e}")
    return None

def process_books_from_txt():
    print("📚 Starting Book Ingestion from book.txt...")
    
    txt_path = config.BASE_DIR / "data" / "raw" / "books" / "book.txt"
    if not txt_path.exists():
        print(f"❌ Raw book.txt not found at {txt_path}")
        return

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    raw_rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(':')
        if len(parts) >= 3:
            title = ":".join(parts[:-2]).strip()
            isbn = parts[-2].strip()
            rating_str = parts[-1].strip()
            try:
                rating = float(rating_str)
            except:
                continue
                
            print(f"Processing: {title} (ISBN: {isbn})")
            
            # Fetch by ISBN first if possible
            book_info = None
            if isbn and isbn.isdigit():
                book_info = fetch_google_books_data(f"isbn:{isbn}")
            
            if not book_info:
                book_info = fetch_google_books_data(f"intitle:{title}")
                
            row = {
                'title': title,
                'isbn': isbn,
                'my_rating': rating,
                'authors': '',
                'publisher': '',
                'publishedDate': '',
                'pageCount': 0,
                'categories': '',
                'averageRating': 0.0,
                'ratingsCount': 0,
                'description': '',
                'thumbnail': '',
                'infoLink': ''
            }
            
            if book_info:
                row['authors'] = ", ".join(book_info.get('authors', []))
                row['publisher'] = book_info.get('publisher') or ''
                row['publishedDate'] = book_info.get('publishedDate') or ''
                row['pageCount'] = book_info.get('pageCount') or 0
                row['categories'] = ", ".join(book_info.get('categories', []))
                row['averageRating'] = book_info.get('averageRating') or 0.0
                row['ratingsCount'] = book_info.get('ratingsCount') or 0
                row['description'] = book_info.get('description') or ''
                images = book_info.get('imageLinks', {})
                row['thumbnail'] = images.get('thumbnail') or images.get('smallThumbnail') or ''
                row['infoLink'] = book_info.get('infoLink') or ''
            
            raw_rows.append(row)
            time.sleep(0.2)
            
    df = pd.DataFrame(raw_rows)
    
    # Save to raw directory
    config.BOOKS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.BOOKS_RAW_DIR / "books_data.csv", index=False)
    
    # Save to processed directory (since we fetched all details during parsing)
    config.BOOKS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.BOOKS_ENRICHED_DATA_PATH, index=False)
    print(f"✅ Enrichment complete. Saved {len(df)} books to {config.BOOKS_ENRICHED_DATA_PATH}")

if __name__ == "__main__":
    process_books_from_txt()
