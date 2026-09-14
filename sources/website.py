# sources/website.py
#
# Plain HTTP fetching and HTML parsing, no browser. This used crawl4ai,
# which drives a headless Chromium that Render's build never installed
# (every crawl failed with "Executable doesn't exist"), and on a 512MB
# instance shared with the whole API a real browser would OOM the web
# process. The trade-off: pages that only render their content with
# JavaScript come back empty.

import time
import uuid
from collections import deque
from urllib.parse import urldefrag, urljoin, urlsplit

import requests
import sentry_sdk
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from config import MAX_CHUNKS_PER_INGEST
from sources.url_guard import assert_public_http_url, safe_get


# ── URL patterns to skip — low-value pages that add noise ──
SKIP_URL_PATTERNS = [
    "?redirected",
    "/author/",
    "/tag/",
    "/category/",
    "/page/",
    "?page=",
    "/feed",
    "/wp-",
    "/cdn-cgi/",
    "?redirectedForLogin",
    "?redirectedForSignup",
    "/magazine/",
    "/embed-guide",
]

SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".mov", ".webm", ".mp3", ".wav", ".zip", ".rar", ".gz",
    ".css", ".js", ".json", ".xml", ".rss", ".doc", ".docx", ".xls",
    ".xlsx", ".ppt", ".pptx", ".csv", ".txt", ".woff", ".woff2", ".ttf",
)

MAX_DEPTH = 3
REQUEST_TIMEOUT = (5, 15)          # (connect, read) per request
# The crawl runs inside the /sources/add request, so it needs a hard stop.
CRAWL_DEADLINE_SECONDS = 60
# Checked against DECOMPRESSED bytes, so a gzip bomb can't blow past it.
MAX_PAGE_BYTES = 5 * 1024 * 1024
# Below this a page is almost always an empty JS shell, not content.
MIN_PAGE_TEXT_CHARS = 50

# Many small-business hosts sit behind bot protection that rejects the
# default python-requests user agent outright.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
    "Accept-Language": "en",
}

# nav is pure link lists repeated on every page. header/footer are kept on
# purpose: that is where a business's address, phone and hours usually live.
NON_CONTENT_TAGS = ["script", "style", "noscript", "template", "svg", "canvas", "iframe", "nav", "form"]

BLOCK_TAGS = [
    "p", "div", "section", "article", "main", "header", "footer", "aside",
    "li", "ul", "ol", "dl", "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "tr", "td", "th", "blockquote", "pre", "address", "figcaption",
    "br", "hr",
]


def normalize_url(url: str) -> str:
    """Strip fragment and trailing slash for deduplication."""
    url, _ = urldefrag(url)
    return url.rstrip("/")


def should_skip(url: str) -> bool:
    """Return True if URL matches a low-value pattern."""
    return any(pattern in url for pattern in SKIP_URL_PATTERNS)


def _site_key(url: str) -> str:
    # example.com and www.example.com are the same site for crawl scope.
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _fetch_html(session, url: str, deadline: float):
    """Returns (final_url, body_bytes, declared_encoding), or None when the
    response isn't a usable HTML page. Every redirect hop is SSRF-checked
    by safe_get, which raises ValueError if one points somewhere internal."""
    res = safe_get(session, url, timeout=REQUEST_TIMEOUT, stream=True)
    try:
        if res.status_code != 200:
            return None
        content_type = res.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type:
            return None

        body = bytearray()
        for chunk in res.iter_content(chunk_size=64 * 1024):
            body.extend(chunk)
            # The read timeout is per-chunk, so a server dripping bytes could
            # otherwise hold this loop open far past the crawl deadline.
            if len(body) >= MAX_PAGE_BYTES or time.monotonic() > deadline:
                break

        encoding = res.encoding if "charset=" in content_type else None
        return res.url, bytes(body[:MAX_PAGE_BYTES]), encoding
    finally:
        res.close()


def _extract(html: bytes, encoding, base_url: str):
    """Returns (text, absolute_links) for one page."""
    soup = BeautifulSoup(html, "lxml", from_encoding=encoding)

    # Links first: the nav removed below is where most internal links are.
    links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)]

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"})
    description = (meta.get("content") or "").strip() if meta else ""

    for tag in soup(NON_CONTENT_TAGS):
        tag.decompose()

    body = soup.body or soup
    # Newlines only at block boundaries. get_text("\n") would split
    # "Open <b>daily</b> 9-5" into three lines and fragment every sentence.
    for tag in body.find_all(BLOCK_TAGS):
        tag.insert_after("\n")
    lines = (" ".join(line.split()) for line in body.get_text().splitlines())
    text = "\n".join(line for line in lines if line)

    parts = [f"# {title}" if title else "", description, text]
    return "\n\n".join(p for p in parts if p), links


def crawl_website(url: str, full_site: bool = True, max_pages: int = 30) -> list[dict]:
    # Nothing validated this URL before it reached the fetcher: any scheme
    # and any host was fetched server-side and indexed into the project's
    # chatbot. file:// could read this server's own .env (service-role key,
    # OpenAI key), and 169.254.169.254 reaches cloud metadata.
    url = url.strip()
    assert_public_http_url(url)

    deadline = time.monotonic() + CRAWL_DEADLINE_SECONDS
    # Failed and non-HTML fetches don't produce pages, so pages alone
    # can't bound how many requests one crawl makes.
    max_fetches = max_pages * 3 if full_site else 1

    queue = deque([(url, 0)])
    queued = {normalize_url(url)}
    indexed = set()
    pages = []
    site = None
    fetches = failures = unexpected = 0
    first_unexpected = None
    stopped_by_deadline = False

    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    try:
        while queue and len(pages) < max_pages and fetches < max_fetches:
            if time.monotonic() > deadline:
                stopped_by_deadline = True
                break

            current, depth = queue.popleft()
            fetches += 1
            try:
                fetched = _fetch_html(session, current, deadline)
                if fetched is None:
                    continue
                final_url, html, encoding = fetched
                text, links = _extract(html, encoding, final_url)
            except (requests.RequestException, ValueError):
                # Routine for real websites (timeouts, resets, redirects to
                # blocked addresses) and not our bug, so not a Sentry event.
                failures += 1
                continue
            except Exception as e:
                # Per-page loop: count here, report once after the loop.
                unexpected += 1
                first_unexpected = first_unexpected or e
                continue

            # Scope comes from where the START page actually landed, so a
            # site that redirects example.com -> www.example.com still crawls.
            if site is None:
                site = _site_key(final_url)
            elif _site_key(final_url) != site:
                continue

            norm = normalize_url(final_url)
            if norm in indexed:
                continue
            indexed.add(norm)
            queued.add(norm)

            if len(text) >= MIN_PAGE_TEXT_CHARS:
                pages.append({"url": final_url, "text": text})

            if not full_site or depth >= MAX_DEPTH:
                continue

            for link in links:
                if len(queue) >= max_fetches:
                    break
                link, _ = urldefrag(link)
                parts = urlsplit(link)
                if parts.scheme not in ("http", "https"):
                    continue
                if _site_key(link) != site:
                    continue
                if parts.path.lower().endswith(SKIP_EXTENSIONS) or should_skip(link):
                    continue
                link_norm = normalize_url(link)
                if link_norm in queued:
                    continue
                queued.add(link_norm)
                queue.append((link, depth + 1))
    finally:
        session.close()

    if unexpected:
        sentry_sdk.capture_message(
            f"Website crawl of {url} hit {unexpected} unexpected error(s); "
            f"first: {type(first_unexpected).__name__}: {first_unexpected}",
            level="warning",
        )

    print(
        f"Crawled {len(pages)} pages from {url} (limit: {max_pages}, "
        f"fetches: {fetches}, failed: {failures}, unexpected: {unexpected}"
        f"{', stopped at deadline' if stopped_by_deadline else ''})"
    )
    return pages


def sync_website(
    url: str,
    project_id: str,
    source_id: str,
    qdrant,
    embeddings,
    collection: str,
    full_site: bool = True,
    max_pages: int = 30,          # FIX: default reduced from 50 to 30
):
    # The purge deliberately happens after a successful crawl (see below).
    # Deleting first meant a site that was temporarily down, rate-limiting
    # us, or blocking the crawler wiped the working index and left the
    # chatbot with nothing.
    pages = crawl_website(url, full_site=full_site, max_pages=max_pages)

    if not pages:
        print("No content found.")
        return {"pages_indexed": 0, "chunks_indexed": 0}

    # FIX: larger chunk size for website content
    # 2000 chars ≈ half the chunks vs 1000, better for long-form articles
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    all_chunks = []
    all_metas = []

    for page in pages:
        chunks = splitter.split_text(page["text"])
        for c in chunks:
            all_chunks.append(c)
            all_metas.append({
                "project_id": project_id,
                "source_id": source_id,
                "source_type": "website",
                "page_url": page["url"],
                "text": c,
            })

    if not all_chunks:
        return {"pages_indexed": len(pages), "chunks_indexed": 0}

    # One unbounded embed call for a large documentation site is an
    # unbounded bill on our own OpenAI key.
    if len(all_chunks) > MAX_CHUNKS_PER_INGEST:
        print(f"website {url} truncated to {MAX_CHUNKS_PER_INGEST} chunks")
        all_chunks = all_chunks[:MAX_CHUNKS_PER_INGEST]
        all_metas = all_metas[:MAX_CHUNKS_PER_INGEST]

    vectors = embeddings.embed_documents(all_chunks)

    # Safe to drop the old index only now that replacement content exists.
    qdrant.delete(
        collection_name=collection,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id)
            )]
        )
    )

    qdrant.upload_points(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=v,
                payload=m
            ) for v, m in zip(vectors, all_metas)
        ]
    )

    print(f"Indexed {len(all_chunks)} chunks from {len(pages)} pages")
    return {"pages_indexed": len(pages), "chunks_indexed": len(all_chunks)}
