"""Headless-browser fallback for pages that only render with JavaScript.

The crawler (sources/website.py) fetches every page with plain HTTP first:
it's ~10x faster and costs almost no memory. Only a page that comes back as
an empty JavaScript shell (React/Vue/Wix-style sites) is rendered here in a
real Chromium, and once a site's start page needs that, the rest of the
site is rendered straight away (its HTML links are empty too).

Only available where Chromium is installed and PLAYWRIGHT_ENABLED is set:
the indexing worker (Dockerfile). On the 512MB main backend (where the
fallback job runner may run) it stays off and crawls are HTML-only, as
before — a browser there is what used to get the instance killed.
"""
import os
from urllib.parse import urlsplit

from sources.url_guard import is_public_http_url

RENDER_TIMEOUT_MS = 20_000
# Content often arrives via XHR after "load"; wait briefly for the network
# to go quiet, but never let a chatty page (analytics, websockets) hang us.
NETWORK_IDLE_TIMEOUT_MS = 5_000
# Images, fonts and media carry no text; skipping them is most of the speed
# and memory saving.
_BLOCKED_RESOURCES = {"image", "media", "font"}


def browser_available() -> bool:
    if os.getenv("PLAYWRIGHT_ENABLED", "").lower() not in ("1", "true", "yes"):
        return False
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return True


class BrowserRenderer:
    """One Chromium for a whole crawl (starting one costs ~1s and ~150MB),
    one fresh page per URL. Use as a context manager; start is lazy, so a
    crawl that never needs JavaScript never launches a browser."""

    def __init__(self, user_agent: str):
        self._user_agent = user_agent
        self._pw = self._browser = self._context = None
        self._host_ok = {}
        self.rendered = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _start(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=True,
            # Cloud Run's /dev/shm is tiny; without this Chromium crashes
            # on larger pages.
            args=["--disable-dev-shm-usage", "--disable-gpu"],
        )
        self._context = self._browser.new_context(
            user_agent=self._user_agent,
            java_script_enabled=True,
            service_workers="block",
        )
        self._context.route("**/*", self._guard)

    def _guard(self, route):
        """Every request the page makes — scripts, XHR, redirects, iframes —
        goes through the same SSRF check as plain fetching. A browser loads
        whatever the page tells it to, so a hostile page could otherwise
        point it at 169.254.169.254 or an internal service."""
        req = route.request
        if req.resource_type in _BLOCKED_RESOURCES:
            return route.abort()
        if req.url.startswith(("data:", "blob:")):
            return route.continue_()
        host = (urlsplit(req.url).hostname or "").lower()
        ok = self._host_ok.get(host)
        if ok is None:
            ok = self._host_ok[host] = is_public_http_url(req.url)
        return route.continue_() if ok else route.abort()

    def render(self, url: str):
        """Returns (final_url, html_bytes), or None if it can't be rendered."""
        if self._browser is None:
            self._start()
        page = self._context.new_page()
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=RENDER_TIMEOUT_MS)
            if response is None or response.status >= 400:
                return None
            try:
                page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
            except Exception:
                pass  # still useful: take whatever has rendered by now
            final_url = page.url
            if not is_public_http_url(final_url):
                return None
            self.rendered += 1
            return final_url, page.content().encode("utf-8")
        finally:
            page.close()

    def close(self):
        for closer in (
            lambda: self._context and self._context.close(),
            lambda: self._browser and self._browser.close(),
            lambda: self._pw and self._pw.stop(),
        ):
            try:
                closer()
            except Exception:
                pass
        self._pw = self._browser = self._context = None
