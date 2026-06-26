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


# --------------------------------------------------------------------------- #
# Library ingestion — pull the user's rated library straight from Hardcover.
# This is the single source of truth for the Books domain: titles, ratings and
# all metadata come from one Hardcover `user_books` record, so there is no
# txt/csv input and no fuzzy title/ISBN join to corrupt enrichment.
# --------------------------------------------------------------------------- #

ME_QUERY = "{ me { id username books_count } }"

LIBRARY_QUERY = """
query($uid: Int!, $limit: Int!, $offset: Int!) {
  user_books(where: {user_id: {_eq: $uid}, rating: {_is_null: false}},
             order_by: {id: asc}, limit: $limit, offset: $offset) {
    rating
    status_id
    edition {
      isbn_13
      pages
      release_date
      edition_format
      publisher { name }
      language { language }
    }
    book {
      title
      slug
      release_year
      pages
      rating
      ratings_count
      description
      image { url }
      cached_tags
      contributions { author { name } }
      editions(where: {publisher_id: {_is_null: false}},
               order_by: {users_count: desc}, limit: 1) {
        isbn_13
        publisher { name }
      }
    }
  }
}
"""


def _gql(query, variables=None):
    """POST a GraphQL query to Hardcover; return the parsed `data` dict (or None)."""
    header = _auth_header()
    if not header:
        print("❌ HARDCOVER_API_KEY missing — cannot reach the library API.")
        return None
    try:
        resp = requests.post(HARDCOVER_URL, headers=header,
                             json={"query": query, "variables": variables or {}}, timeout=30)
        payload = resp.json()
    except Exception as e:
        print(f"  ⚠️ Hardcover request error: {e}")
        return None
    if payload.get("errors"):
        print(f"  ⚠️ Hardcover GraphQL errors: {payload['errors']}")
        return None
    return payload.get("data")


def _genres_from_cached_tags(tags):
    """Extract the 'Genre' tag names from a Hardcover cached_tags blob."""
    if not isinstance(tags, dict):
        return []
    return [t.get("tag") for t in tags.get("Genre", []) if t.get("tag")]


def _shape_user_book(ub):
    """Turn one Hardcover user_books record into an enriched-schema row dict.

    Publisher and ISBN are taken from the user's own logged edition first (the most
    accurate source — e.g. "Puffin Books", "Amulet Books"), then from the book's
    most-held edition that has a publisher, so the publisher column is filled from
    Hardcover wherever possible. pageCount stays on book.pages (the canonical count)
    so the model's numeric features are unchanged by this enrichment.
    """
    book = ub.get("book") or {}
    edition = ub.get("edition") or {}
    authors = [c["author"]["name"] for c in book.get("contributions", []) if c.get("author")]
    genres = _genres_from_cached_tags(book.get("cached_tags"))
    book_edition = (book.get("editions") or [{}])
    book_edition = book_edition[0] if book_edition else {}

    # Publisher: user edition -> book's best publisher-bearing edition.
    publisher = ((edition.get("publisher") or {}).get("name")
                 or (book_edition.get("publisher") or {}).get("name") or "")
    # ISBN: user edition -> book edition.
    isbn = edition.get("isbn_13") or book_edition.get("isbn_13") or ""
    language = (edition.get("language") or {}).get("language") or ""
    slug = book.get("slug")
    rating = book.get("rating")
    return {
        "title": book.get("title"),
        "isbn": isbn,
        "my_rating": ub.get("rating"),
        "authors": ", ".join(dict.fromkeys(authors)),
        "publisher": publisher,
        "publishedDate": str(book.get("release_year") or ""),
        "pageCount": book.get("pages") or edition.get("pages") or 0,
        "categories": ", ".join(genres),
        "averageRating": round(rating, 2) if rating else 0.0,
        "ratingsCount": book.get("ratings_count") or 0,
        "description": book.get("description") or "",
        "thumbnail": (book.get("image") or {}).get("url") or "",
        "infoLink": f"https://hardcover.app/books/{slug}" if slug else "",
        "language": language,
    }


# Fields we try to backfill from Open Library when Hardcover leaves them empty.
_OL_BACKFILL = {
    "publisher": "publisher",
    "pageCount": "pageCount",
    "categories": "categories",
    "description": "description",
    "thumbnail": "thumbnail",
    "publishedDate": "publishedDate",
    "authors": "authors",
}


def _is_empty(v):
    return v in (None, "", 0, 0.0, "0", "[]")


def _backfill_from_openlibrary(row):
    """Fill any enriched field Hardcover left empty from Open Library (keyed by ISBN)."""
    gaps = [k for k in _OL_BACKFILL if _is_empty(row.get(k))]
    if not gaps or not row.get("isbn"):
        return row
    ol = fetch_openlibrary(row["isbn"]) or {}
    for field, ol_key in _OL_BACKFILL.items():
        if _is_empty(row.get(field)) and not _is_empty(ol.get(ol_key)):
            row[field] = ol[ol_key]
    return row


def fetch_library_from_hardcover(page_size=50):
    """Fetch the authenticated user's rated library from Hardcover.

    Returns a list of enriched-schema row dicts (one per rated book), in stable
    user_books.id ascending order so the dataset — and therefore the frozen fold
    registry built on top of it — is reproducible run to run.
    """
    me = _gql(ME_QUERY)
    if not me or not me.get("me"):
        print("❌ Could not resolve the current Hardcover user (check the token).")
        return []
    user = me["me"][0]
    uid = user["id"]
    print(f"📚 Hardcover user: {user.get('username')} (id {uid}) — "
          f"{user.get('books_count')} books in library.")

    rows, offset = [], 0
    while True:
        data = _gql(LIBRARY_QUERY, {"uid": uid, "limit": page_size, "offset": offset})
        if data is None:
            break
        batch = data.get("user_books", [])
        for ub in batch:
            row = _shape_user_book(ub)
            if row.get("title") and row.get("my_rating") is not None:
                # Fill any Hardcover gap (publisher, pages, categories, …) from Open Library.
                row = _backfill_from_openlibrary(row)
                rows.append(row)
        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(0.2)  # be polite to the API
    return rows


def _norm_key(series):
    """Normalized join key (lower/stripped) for matching titles across runs."""
    return series.fillna("").astype(str).str.strip().str.lower()


def build_books_from_library(merge=True):
    """Build the Books enriched dataset from the Hardcover library API.

    merge=True (default): keep every existing (hand-cleaned) row in the enriched
    CSV untouched and only fetch+append books that are newly in your library.
    merge=False: overwrite the enriched CSV entirely from the library.
    """
    print("📚 Starting Book Ingestion (Hardcover library API)...")
    if not HARDCOVER_TOKEN:
        print("❌ HARDCOVER_API_KEY not set — the library API requires it. Aborting.")
        return

    rows = fetch_library_from_hardcover()
    if not rows:
        print("❌ No rated books returned from the Hardcover library. Aborting.")
        return

    df = pd.DataFrame(rows)

    if merge and config.BOOKS_ENRICHED_DATA_PATH.exists():
        try:
            existing = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
        except Exception:
            existing = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding="latin1")
        if not existing.empty and "title" in existing.columns:
            have = set(_norm_key(existing["title"]))
            new_only = df[~_norm_key(df["title"]).isin(have)]
            print(f"   Merge: {len(existing)} existing (cleaned) rows preserved, "
                  f"{len(new_only)} new book(s) from library appended.")
            df = pd.concat([existing, new_only], ignore_index=True)

    config.BOOKS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.BOOKS_RAW_DIR / "books_data.csv", index=False)
    config.BOOKS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.BOOKS_ENRICHED_DATA_PATH, index=False)

    filled = (df["authors"].astype(str).str.len() > 0).sum()
    pages = pd.to_numeric(df["pageCount"], errors="coerce").fillna(0).gt(0).sum()
    cats = (df["categories"].astype(str).str.len() > 0).sum()
    pubs = (df["publisher"].astype(str).str.len() > 0).sum()
    print(f"✅ Done. {len(df)} rated books | authors: {filled} | pages: {pages} | "
          f"genres: {cats} | publisher: {pubs}")
    print(f"   Saved to {config.BOOKS_ENRICHED_DATA_PATH}")


if __name__ == "__main__":
    build_books_from_library()
