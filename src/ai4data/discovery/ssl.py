"""TLS trust store setup for discovery HTTP clients (catalog, extract, PDF fetch).

Uses :mod:`truststore` when installed so Python/httpx pick up the OS trust store
(e.g. macOS Keychain with corporate proxy CAs). Idempotent — safe to call multiple times.
"""

from __future__ import annotations

_tls_configured = False


def configure_tls_trust_store() -> None:
    """Inject OS trust store into :mod:`ssl` (no-op if ``truststore`` is not installed)."""
    global _tls_configured
    if _tls_configured:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    _tls_configured = True
