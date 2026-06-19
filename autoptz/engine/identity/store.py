"""Identity persistence adapter (labeled-only) over the SQLite ConfigStore.

Retention policy — **labeled-only persisted**
----------------------------------------------
Only *labeled* (human-named) identities are written to the database; the
auto-harvested "Unlabeled people" tray lives in memory in the
:class:`~autoptz.engine.identity.service.IdentityService` and is intentionally
**not** persisted (it vanishes on restart).  This adapter is therefore a thin
shim that translates the service's :class:`IdentityRecord` objects to/from the
``identities`` / ``identity_embeddings`` tables in
:class:`autoptz.config.store.ConfigStore` (which already owns the schema).

Keeping this layer separate from the service lets the service run with no store
at all (pure in-memory, used by tests and headless harvesting) and lets the
store be swapped without touching gallery logic.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoptz.config.models import IdentityRecord
    from autoptz.config.store import ConfigStore

log = logging.getLogger(__name__)


class IdentityStore:
    """Persist *labeled* identities through a :class:`ConfigStore`.

    A ``None`` config store makes every method a safe no-op (in-memory gallery),
    so the service degrades gracefully when persistence is unavailable.
    """

    def __init__(self, config_store: ConfigStore | None) -> None:
        self._store = config_store

    @property
    def persistent(self) -> bool:
        return self._store is not None

    def load(self) -> list[IdentityRecord]:
        """Load all persisted (labeled) identities; never raises."""
        if self._store is None:
            return []
        try:
            return self._store.load_identities()
        except Exception:  # noqa: BLE001
            log.warning("IdentityStore.load failed", exc_info=True)
            return []

    def save(self, record: IdentityRecord) -> None:
        """Persist *record* — but only if it is labeled (retention policy)."""
        if self._store is None:
            return
        if not getattr(record, "labeled", True):
            # Unlabeled / auto-harvested identities are never written to disk.
            return
        try:
            self._store.save_identity(record)
        except Exception:  # noqa: BLE001
            log.warning("IdentityStore.save failed for %s", record.id, exc_info=True)

    def delete(self, identity_id: str) -> None:
        if self._store is None:
            return
        try:
            self._store.delete_identity(identity_id)
        except Exception:  # noqa: BLE001
            log.warning("IdentityStore.delete failed for %s", identity_id, exc_info=True)

    def delete_photo(self, identity_id: str, index: int) -> bool:
        """Delete one stored candidate photo (``identity_photos`` row) by index.

        A thin pass-through to :meth:`ConfigStore.delete_identity_photo`; a
        ``None`` config store (in-memory gallery) is a safe no-op.  Never raises.
        """
        if self._store is None:
            return False
        try:
            return bool(self._store.delete_identity_photo(identity_id, index))
        except Exception:  # noqa: BLE001
            log.warning(
                "IdentityStore.delete_photo failed for %s[%s]",
                identity_id, index, exc_info=True,
            )
            return False
