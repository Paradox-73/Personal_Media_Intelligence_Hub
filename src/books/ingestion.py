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

# Hardcover stores the full "Bearer <jwt>" string in HARDCOVER_API_KEY
HARDCOVER_TOKEN = os.getenv("HARDCOVER_API_KEY")
HARDCOVER_URL = "https://api.hardcover.app/v1/graphql"
OPENLIBRARY_URL = "https://openlibrary.org/api/books"

HARDCOVER_QUERY = """
query BookByIsbn($isbn: String!) {
  editions(where: {isbn_13: {_eq: $isbn}}, limit: 1) {
    pages
    book {
      title
      slug
      rating
      ratings_count
      release_year
      description
      cached_tags
      contributions { author { name } }
    }
  }
}
"""


def _auth_header():
    if not HARDCOVER_TOKEN:
        return None
    token = HARDCOVER_TOKEN.strip()
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Authorization": token, "Content-Type": "application/json"}


def fetch_hardcover(isbn):
    """Return Hardcover metadata (rating, description, authors, genres) for an ISBN-13."""
    header = _auth_header()
    if not header or not isbn:
        return None
    try:
        resp = requests.post(HARDCOVER_URL, headers=header,
                             json={"query": HARDCOVER_QUERY, "variables": {"isbn": str(isbn)}}, timeout=15)
        editions = (resp.json().get("data", {}) or {}).get("editions", [])
        if not editions:
            return None
        ed = editions[0]
        book = ed.get("book") or {}

        authors = [c["author"]["name"] for c in book.get("contributions", []) if c.get("author")]

        genres = []
        tags = book.get("cached_tags") or {}
        if isinstance(tags, dict):
            genres = [t.get("tag") for t in tags.get("Genre", []) if t.get("tag")]

        slug = book.get("slug")
        return {
            "title": book.get("title"),
            "authors": ", ".join(dict.fromkeys(authors)),
            "pageCount": ed.get("pages") or 0,
            "categories": ", ".join(genres),
            "averageRating": round(book.get("rating"), 2) if book.get("rating") else 0.0,
            "ratingsCount": book.get("ratings_count") or 0,
            "description": book.get("description") or "",
            "publishedDate": str(book.get("release_year") or ""),
            "infoLink": f"https://hardcover.app/books/{slug}" if slug else "",
        }
    except Exception as e:
        print(f"  ⚠️ Hardcover error for {isbn}: {e}")
        return None


def fetch_openlibrary(isbn):
    """Return Open Library metadata (pages, cover, publisher, subjects) for an ISBN."""
    if not isbn:
        return None
    try:
        resp = requests.get(OPENLIBRARY_URL,
                            params={"bibkeys": f"ISBN:{isbn}", "format": "json", "jscmd": "data"}, timeout=12)
        data = resp.json().get(f"ISBN:{isbn}")
        if not data:
            return None
        cover = data.get("cover") or {}
        return {
            "title": data.get("title"),
            "authors": ", ".join(a["name"] for a in data.get("authors", [])),
            "pageCount": data.get("number_of_pages") or 0,
            "categories": ", ".join(s["name"] for s in data.get("subjects", [])[:8]),
            "publisher": ", ".join(p["name"] for p in data.get("publishers", [])),
            "publishedDate": data.get("publish_date") or "",
            "thumbnail": cover.get("medium") or cover.get("large") or cover.get("small") or "",
            "infoLink": data.get("url") or "",
        }
    except Exception as e:
        print(f"  ⚠️ Open Library error for {isbn}: {e}")
        return None


def _coalesce(*vals):
    """First truthy, non-empty value."""
    for v in vals:
        if v not in (None, "", 0, 0.0, "0"):
            return v
    return vals[-1] if vals else ""


OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"


def search_books_by_query(query, max_results=5):
    """Live Open Library search for the Oracle (keyless). Returns lightweight dicts."""
    if not query:
        return []
    try:
        resp = requests.get(OPENLIBRARY_SEARCH_URL, params={
            "q": query, "limit": max_results,
            "fields": "key,title,author_name,first_publish_year,isbn,cover_i,subject",
        }, timeout=12)
        docs = resp.json().get("docs", [])
    except Exception as e:
        print(f"  ⚠️ Open Library search error: {e}")
        return []

    out = []
    for d in docs[:max_results]:
        isbns = d.get("isbn") or []
        cover_i = d.get("cover_i")
        out.append({
            # Prefer an ISBN as the id (enables rich Hardcover/OL detail fetch);
            # fall back to the OL work key.
            "id": isbns[0] if isbns else d.get("key", ""),
            "title": d.get("title", "Unknown"),
            "author": ", ".join(d.get("author_name", [])[:2]) or "Unknown",
            "year": d.get("first_publish_year", ""),
            "subjects": d.get("subject", [])[:8],
            "poster_path": f"https://covers.openlibrary.org/b/id/{cover_i}-M.jpg" if cover_i else None,
        })
    return out


def fetch_book_details_by_id(book_id, fallback=None):
    """Fetch full detail for a selected book, shaped for the Oracle transform.

    `book_id` is an ISBN (preferred) or an Open Library work key. `fallback` is the
    search-result dict, used to backfill fields the detail APIs omit.
    """
    fallback = fallback or {}
    data = {}

    looks_like_isbn = str(book_id).replace("-", "").isdigit()
    if looks_like_isbn:
        hc = fetch_hardcover(book_id) or {}
        ol = fetch_openlibrary(book_id) or {}
        data = {
            "title": _coalesce(hc.get("title"), ol.get("title"), fallback.get("title")),
            "authors": _coalesce(hc.get("authors"), ol.get("authors"), fallback.get("author"), ""),
            "categories": _coalesce(hc.get("categories"), ol.get("categories"),
                                    ", ".join(fallback.get("subjects", [])), ""),
            "description": _coalesce(hc.get("description"), ""),
            "pageCount": _coalesce(hc.get("pageCount"), ol.get("pageCount"), 0),
            "averageRating": _coalesce(hc.get("averageRating"), 0.0),
            "ratingsCount": _coalesce(hc.get("ratingsCount"), 0),
            "publishedDate": _coalesce(hc.get("publishedDate"), ol.get("publishedDate"),
                                       str(fallback.get("year", ""))),
            "thumbnail": _coalesce(ol.get("thumbnail"), fallback.get("poster_path"), ""),
            "infoLink": _coalesce(hc.get("infoLink"), ol.get("infoLink"), ""),
        }
    else:
        # Open Library work key -> work JSON (description + subjects only)
        try:
            resp = requests.get(f"https://openlibrary.org{book_id}.json", timeout=12)
            w = resp.json()
            desc = w.get("description")
            if isinstance(desc, dict):
                desc = desc.get("value", "")
            data = {
                "title": _coalesce(w.get("title"), fallback.get("title")),
                "authors": fallback.get("author", ""),
                "categories": ", ".join(w.get("subjects", fallback.get("subjects", []))[:8]),
                "description": desc or "",
                "pageCount": 0,
                "averageRating": 0.0,
                "ratingsCount": 0,
                "publishedDate": str(fallback.get("year", "")),
                "thumbnail": fallback.get("poster_path", ""),
                "infoLink": f"https://openlibrary.org{book_id}",
            }
        except Exception as e:
            print(f"  ⚠️ Open Library work fetch error: {e}")
            return None

    if not data.get("title"):
        return None
    data["year"] = data.get("publishedDate")
    data["genre"] = [c.strip() for c in str(data.get("categories", "")).split(",") if c.strip()]
    data["author"] = data.get("authors")
    data["processing_status"] = "success"
    return data


def process_books_from_txt():
    print("📚 Starting Book Ingestion (Hardcover + Open Library)...")
    if not HARDCOVER_TOKEN:
        print("⚠️ HARDCOVER_API_KEY not set — will rely on Open Library only.")

    txt_path = config.BOOKS_RAW_DIR / "book.txt"
    if not txt_path.exists():
        print(f"❌ Raw book.txt not found at {txt_path}")
        return

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Format: "Title : ISBN : rating"
        parts = line.split(":")
        if len(parts) < 3:
            continue
        title = ":".join(parts[:-2]).strip()
        isbn = parts[-2].strip()
        try:
            rating = float(parts[-1].strip())
        except ValueError:
            continue

        print(f"Processing: {title} (ISBN: {isbn})")
        hc = fetch_hardcover(isbn) or {}
        ol = fetch_openlibrary(isbn) or {}

        rows.append({
            "title": title,
            "isbn": isbn,
            "my_rating": rating,
            "authors": _coalesce(hc.get("authors"), ol.get("authors"), ""),
            "publisher": _coalesce(ol.get("publisher"), ""),
            "publishedDate": _coalesce(hc.get("publishedDate"), ol.get("publishedDate"), ""),
            "pageCount": _coalesce(hc.get("pageCount"), ol.get("pageCount"), 0),
            "categories": _coalesce(hc.get("categories"), ol.get("categories"), ""),
            "averageRating": _coalesce(hc.get("averageRating"), 0.0),
            "ratingsCount": _coalesce(hc.get("ratingsCount"), 0),
            "description": _coalesce(hc.get("description"), ""),
            "thumbnail": _coalesce(ol.get("thumbnail"), ""),
            "infoLink": _coalesce(hc.get("infoLink"), ol.get("infoLink"), ""),
        })
        time.sleep(0.3)  # be polite to both APIs

    df = pd.DataFrame(rows)

    config.BOOKS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.BOOKS_RAW_DIR / "books_data.csv", index=False)
    config.BOOKS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.BOOKS_ENRICHED_DATA_PATH, index=False)

    filled = (df["authors"].astype(str).str.len() > 0).sum()
    pages = pd.to_numeric(df["pageCount"], errors="coerce").fillna(0).gt(0).sum()
    print(f"✅ Done. {len(df)} books | authors filled: {filled} | pages filled: {pages}")
    print(f"   Saved to {config.BOOKS_ENRICHED_DATA_PATH}")


if __name__ == "__main__":
    process_books_from_txt()
