import pandas as pd
import json
import os
import argparse
import sys
import requests
import numpy as np
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# Load Environment Variables
load_dotenv()

# Google Books API doesn't strictly require a key for basic search, 
# but it's better to have one. If missing, it still works with limited quota.
GOOGLE_BOOKS_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

def is_effectively_empty(val):
    if val is None: return True
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
    try:
        if pd.isna(val): return True
    except:
        pass
    s = str(val).strip().lower()
    return s in ['', 'nan', 'n/a', 'none', 'null', 'unknown']

def fetch_google_books_data(title, author=None):
    query = f"intitle:{title}"
    if author:
        query += f"+inauthor:{author}"
    
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    if GOOGLE_BOOKS_KEY:
        url += f"&key={GOOGLE_BOOKS_KEY}"
        
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('totalItems', 0) > 0:
            # Take the first result
            item = data['items'][0]['volumeInfo']
            return item
    except Exception as e:
        print(f"Error fetching from Google Books: {e}")
    return None

def enrich_book_data():
    print("📚 Starting Book Data Enrichment...")
    
    raw_path = config.BOOKS_RAW_DIR / "books_data.csv"
    if not raw_path.exists():
        print(f"❌ Raw data not found at {raw_path}")
        return

    try:
        df = pd.read_csv(raw_path)
    except:
        df = pd.read_csv(raw_path, encoding='latin1')
        
    print(f"📊 Loaded {len(df)} books from raw data.")

    enriched_rows = []
    
    for idx, row in df.iterrows():
        title = row['title']
        author = row.get('authors')
        print(f"[{idx+1}/{len(df)}] Processing: {title}")
        
        # Check if we need to fetch more data
        # We fetch if critical fields like description or categories are missing
        needs_fetch = is_effectively_empty(row.get('description')) or is_effectively_empty(row.get('categories'))
        
        if needs_fetch:
            book_info = fetch_google_books_data(title, author)
            if book_info:
                # Fill missing columns
                if is_effectively_empty(row.get('authors')):
                    row['authors'] = ", ".join(book_info.get('authors', []))
                if is_effectively_empty(row.get('publisher')):
                    row['publisher'] = book_info.get('publisher')
                if is_effectively_empty(row.get('publishedDate')):
                    row['publishedDate'] = book_info.get('publishedDate')
                if is_effectively_empty(row.get('pageCount')):
                    row['pageCount'] = book_info.get('pageCount')
                if is_effectively_empty(row.get('categories')):
                    row['categories'] = ", ".join(book_info.get('categories', []))
                if is_effectively_empty(row.get('averageRating')):
                    row['averageRating'] = book_info.get('averageRating')
                if is_effectively_empty(row.get('ratingsCount')):
                    row['ratingsCount'] = book_info.get('ratingsCount')
                if is_effectively_empty(row.get('description')):
                    row['description'] = book_info.get('description')
                if is_effectively_empty(row.get('thumbnail')):
                    images = book_info.get('imageLinks', {})
                    row['thumbnail'] = images.get('thumbnail') or images.get('smallThumbnail')
                if is_effectively_empty(row.get('infoLink')):
                    row['infoLink'] = book_info.get('infoLink')
            
            # Throttle API calls
            time.sleep(0.2)

        enriched_rows.append(row.to_dict())

    df_enriched = pd.DataFrame(enriched_rows)
    
    # Save to processed directory
    config.BOOKS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(config.BOOKS_ENRICHED_DATA_PATH, index=False)
    print(f"✅ Enrichment complete. Saved to {config.BOOKS_ENRICHED_DATA_PATH}")

if __name__ == "__main__":
    enrich_book_data()
