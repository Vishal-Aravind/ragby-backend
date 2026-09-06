"""Shared guard for user-supplied URLs that this server will fetch.

Any endpoint that takes a URL from a user and then requests it server-side
can be used to reach things the user cannot reach themselves: cloud metadata
endpoints, internal admin panels, localhost services, or — if the scheme
isn't constrained — local files. The database source already had a check
like this; the website crawler had none at all.
"""
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit


def assert_safe_host(host: str):
    """Resolve a hostname and reject it if any address it maps to is
    internal. Practical mitigation, not a complete defense against
    DNS-rebinding (the name is resolved here and again by the fetcher)."""
    if not host:
        raise ValueError("Could not parse a host from this URL.")

    try:
        addrinfo = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ValueError("Could not resolve this host.")

    for _, _, _, _, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(
                "That address isn't reachable — internal or private network "
                "addresses aren't allowed."
            )


def assert_public_http_url(url: str):
    """Validate a URL the crawler is about to fetch.

    The scheme allowlist matters as much as the IP check: without it,
    `file:///app/.env` would be read off this server's own disk and indexed
    into a chatbot, exposing every key in the environment.
    """
    if not url or not url.strip():
        raise ValueError("Please enter a website address.")

    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError("Only http:// and https:// addresses can be indexed.")

    assert_safe_host(parts.hostname)


def is_public_http_url(url: str) -> bool:
    """Non-raising form of assert_public_http_url, for filtering a list of
    already-fetched URLs rather than gating a single request."""
    try:
        assert_public_http_url(url)
        return True
    except ValueError:
        return False


def safe_get(session, url: str, max_redirects: int = 5, timeout: int = 30):
    """GET a URL, validating EVERY hop instead of only the first.

    Validating just the entered URL is not enough on its own: a public
    address that 302s to an internal one passes the front-door check and
    then fetches the internal resource anyway. Redirects are disabled and
    followed manually so each destination is re-validated before we go
    there."""
    current = url
    for _ in range(max_redirects + 1):
        assert_public_http_url(current)
        res = session.get(current, allow_redirects=False, timeout=timeout)
        if res.status_code not in (301, 302, 303, 307, 308):
            return res
        location = res.headers.get("Location")
        if not location:
            return res
        current = urljoin(current, location)
    raise ValueError("That link redirected too many times.")
