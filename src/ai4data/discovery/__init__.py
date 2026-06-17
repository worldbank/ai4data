"""Discovery-facing integrations (e.g. NADA catalog HTTP and batch jobs).

Filesystem layout for discovery caches and id lists: :mod:`ai4data.discovery.paths`.
Catalog API URLs and embedding-template root: :mod:`ai4data.discovery.config`.
"""

from .ssl import configure_tls_trust_store

configure_tls_trust_store()
