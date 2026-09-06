# sources/website.py

import uuid
import asyncio
from urllib.parse import urlparse, urldefrag
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DomainFilter, FilterChain


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


from config import MAX_CHUNKS_PER_INGEST
from sources.url_guard import assert_public_http_url


def get_domain(url: str) -> str:
    return urlparse(url).netloc


def normalize_url(url: str) -> str:
    """Strip fragment and trailing slash for deduplication."""
    url, _ = urldefrag(url)
    return url.rstrip("/")


def should_skip(url: str) -> bool:
    """Return True if URL matches a low-value pattern."""
    return any(pattern in url for pattern in SKIP_URL_PATTERNS)


async def _crawl(url: str, full_site: bool, max_pages: int) -> list[dict]:
    # Nothing validated this URL before it reached the browser: any scheme
    # and any host was fetched server-side and indexed into the project's
    # chatbot. file:// could read this server's own .env (service-role key,
    # OpenAI key), and 169.254.169.254 reaches cloud metadata.
    assert_public_http_url(url)

    pages = []
    seen_urls = set()

    async with AsyncWebCrawler() as crawler:
        if not full_site:
            config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            result = await crawler.arun(url=url, config=config)
            if result.success and result.markdown:
                pages.append({"url": url, "text": result.markdown.strip()})

        else:
            domain = get_domain(url)
            strategy = BFSDeepCrawlStrategy(
                max_depth=3,
                max_pages=max_pages,
                filter_chain=FilterChain([
                    DomainFilter(allowed_domains=[domain])
                ]),
            )
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                deep_crawl_strategy=strategy,
            )
            results = await crawler.arun(url=url, config=config)
            result_list = results if isinstance(results, list) else [results]

            for r in result_list:
                if not r.success or not r.markdown:
                    continue

                # Skip low-value URLs
                if should_skip(r.url):
                    print(f"Skipped: {r.url}")
                    continue

                # Deduplicate
                norm = normalize_url(r.url)
                if norm in seen_urls:
                    continue
                seen_urls.add(norm)

                # Enforce strict page limit
                if len(pages) >= max_pages:
                    break

                pages.append({"url": r.url, "text": r.markdown.strip()})

    print(f"Crawled {len(pages)} pages from {url} (limit: {max_pages})")
    return pages


def crawl_website(url: str, full_site: bool = True, max_pages: int = 30) -> list[dict]:
    return asyncio.run(_crawl(url, full_site, max_pages))


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