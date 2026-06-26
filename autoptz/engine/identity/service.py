"""IdentityService: the in-memory face gallery with labeled-only persistence.

Responsibilities
----------------
- Hold every identity in memory: both **labeled** (named, persisted) and
  **unlabeled** (auto-harvested "Person N", memory-only) records.
- CRUD the gallery:
    * :meth:`enroll`           — add a labeled identity from an embedding + thumb.
    * :meth:`add_unlabeled`    — create a memory-only "Person N" (auto-harvest).
    * :meth:`label`            — promote an unlabeled identity → named + enabled
                                 + persisted.
    * :meth:`rename` / :meth:`delete` / :meth:`set_enabled`.
    * :meth:`merge`            — fold one identity's embeddings + thumbnail into
                                 another (e.g. de-duplicating two harvested faces).
- Score embeddings against the gallery (cosine over a per-identity template set)
  to power :class:`~autoptz.engine.pipeline.identify.FaceRecognizer.match`.
- A monotonically increasing :attr:`version` bumps on every mutation so other
  components (the worker's face stack, a future cache) can do a cheap
  versioned reload instead of polling contents.

Retention policy — **labeled-only persisted**
----------------------------------------------
Labeled identities are written to the DB via :class:`IdentityStore`; unlabeled
ones stay in RAM and are dropped on restart.  ``label()`` is the single point
that flips an identity from memory-only to persisted.

Thread-safety
-------------
A re-entrant lock guards the gallery so the camera worker thread (harvesting +
matching) and the GUI thread (CRUD via EngineClient) can share one service.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from autoptz.config.models import IdentityRecord
from autoptz.engine.identity.store import IdentityStore
from autoptz.engine.pipeline.identify import (
    cosine,
    embedding_from_bytes,
    embedding_to_bytes,
    normalize,
)

if TYPE_CHECKING:
    from autoptz.config.store import ConfigStore

log = logging.getLogger(__name__)

# Cap embeddings per identity so a long auto-harvest session can't grow a
# template set without bound; keep the most recent.
_MAX_EMBEDDINGS_PER_IDENTITY = 16

# Cap candidate profile photos per identity (the recognition crops the user can
# pick a profile from); keep the most recent few.
_MAX_THUMBNAILS_PER_IDENTITY = 8


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class IdentityService:
    """In-memory gallery with labeled-only persistence (see module docstring)."""

    def __init__(
        self,
        config_store: ConfigStore | None = None,
        *,
        store: IdentityStore | None = None,
    ) -> None:
        self._store = store or IdentityStore(config_store)
        self._lock = threading.RLock()
        # id → IdentityRecord (labeled + unlabeled live together here)
        self._records: dict[str, IdentityRecord] = {}
        # id → list[np.float32] normalised templates (parallel cache for fast cosine)
        self._templates: dict[str, list[NDArray[np.float32]]] = {}
        self._version = 0
        self._auto_counter = 0
        self.reload()

    # ── versioned reload ─────────────────────────────────────────────────────────

    @property
    def version(self) -> int:
        return self._version

    def _bump(self) -> None:
        self._version += 1

    def reload(self) -> None:
        """Reload labeled identities from the store, preserving unlabeled ones."""
        with self._lock:
            unlabeled = {rid: rec for rid, rec in self._records.items() if not rec.labeled}
            self._records = {}
            self._templates = {}
            for rec in self._store.load():
                self._index(rec)
            # Unlabeled (memory-only) records survive a reload.
            for rid, rec in unlabeled.items():
                self._records[rid] = rec
                self._templates[rid] = [embedding_from_bytes(b) for b in rec.embeddings]
            self._bump()

    def _index(self, rec: IdentityRecord) -> None:
        self._records[rec.id] = rec
        self._templates[rec.id] = [embedding_from_bytes(b) for b in rec.embeddings]

    # ── reads ────────────────────────────────────────────────────────────────────

    def all_identities(self) -> list[IdentityRecord]:
        with self._lock:
            return list(self._records.values())

    def labeled_identities(self) -> list[IdentityRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.labeled]

    def unlabeled_identities(self) -> list[IdentityRecord]:
        with self._lock:
            return [r for r in self._records.values() if not r.labeled]

    def enabled_identities(self) -> list[IdentityRecord]:
        """Identities the engine should actively match (enabled + has templates)."""
        with self._lock:
            return [r for r in self._records.values() if r.enabled and self._templates.get(r.id)]

    def matchable_identities(self) -> list[IdentityRecord]:
        """Every identity with templates — enabled or not, labeled or not.

        Used for *recognition* (and auto-harvest de-duplication): a harvested
        "Person N" is created disabled, but we still want the next frame to
        recognise the same face and bind it to the existing record instead of
        spawning a duplicate.  ``enabled`` only gates auto-*following*, not
        recognition.
        """
        with self._lock:
            return [r for r in self._records.values() if self._templates.get(r.id)]

    def get(self, identity_id: str) -> IdentityRecord | None:
        with self._lock:
            return self._records.get(identity_id)

    def best_score(
        self,
        identity_id: str,
        embedding: NDArray[np.floating],
    ) -> float:
        """Max cosine similarity of *embedding* against an identity's templates."""
        with self._lock:
            templates = self._templates.get(identity_id)
        if not templates:
            return 0.0
        q = normalize(embedding)
        return max(cosine(q, t) for t in templates)

    # ── mutations ────────────────────────────────────────────────────────────────

    def enroll(
        self,
        name: str,
        embedding: NDArray[np.floating] | None,
        thumbnail: bytes | None = None,
        *,
        identity_id: str | None = None,
        enabled: bool = True,
    ) -> IdentityRecord:
        """Add a new *labeled* identity (persisted).  ``embedding`` may be None."""
        embeddings = [embedding_to_bytes(embedding)] if embedding is not None else []
        # Pass ``id`` only when supplied (else IdentityRecord's default factory
        # generates one).  Typed ``dict[str, Any]`` so the ** unpack doesn't make
        # mypy infer every field as ``str``.
        id_kwargs: dict[str, Any] = {"id": identity_id} if identity_id else {}
        rec = IdentityRecord(
            **id_kwargs,
            name=name,
            embeddings=embeddings,
            thumbnail=thumbnail,
            enabled=enabled,
            labeled=True,
        )
        with self._lock:
            self._index(rec)
            self._store.save(rec)
            self._bump()
        return rec

    def add_unlabeled(
        self,
        embedding: NDArray[np.floating] | None,
        thumbnail: bytes | None = None,
    ) -> IdentityRecord:
        """Create an in-memory **unlabeled** identity ("Person N").

        Memory-only by the retention policy — never written to the DB until it
        is promoted by :meth:`label`.  Created disabled so it does not compete
        for the target until a human names it.
        """
        with self._lock:
            self._auto_counter += 1
            name = f"Person {self._auto_counter}"
            embeddings = [embedding_to_bytes(embedding)] if embedding is not None else []
            rec = IdentityRecord(
                name=name,
                embeddings=embeddings,
                thumbnail=thumbnail,
                thumbnails=[thumbnail] if thumbnail else [],
                enabled=False,
                labeled=False,
            )
            self._index(rec)
            self._bump()
        return rec

    def ingest_record(self, record: IdentityRecord) -> bool:
        """Index a pre-built identity from another process, in-memory + idempotent.

        Used by the opt-in process-per-camera relay: an unlabeled "Person N"
        harvested in one child is broadcast to the others so the same face is
        matchable everywhere (labeled identities already converge via the shared
        DB).  Preserves the record's id (cross-process identity) and bumps the
        version so each worker's versioned reload picks it up.  Memory-only —
        never persisted here (avoids double-writing labeled records the DB already
        owns).  Returns True iff the gallery actually changed.
        """
        with self._lock:
            held = self._records.get(record.id)
            if held is not None and held.updated_at >= record.updated_at:
                return False  # we already have this (or a newer) copy
            self._index(record)
            self._bump()
            return True

    def add_embedding(
        self,
        identity_id: str,
        embedding: NDArray[np.floating],
        thumbnail: bytes | None = None,
    ) -> bool:
        """Append an embedding (and optionally a candidate photo) to an identity.

        Both sets are bounded (most recent kept).  Passing a fresh ``thumbnail``
        as a person stays on camera accrues a few varied recognition shots the
        user can later pick a profile photo from.
        """
        with self._lock:
            rec = self._records.get(identity_id)
            if rec is None:
                return False
            blobs = (list(rec.embeddings) + [embedding_to_bytes(embedding)])[
                -_MAX_EMBEDDINGS_PER_IDENTITY:
            ]
            update: dict[str, object] = {"embeddings": blobs, "updated_at": _now()}
            if thumbnail is not None:
                photos = (list(rec.thumbnails) + [thumbnail])[-_MAX_THUMBNAILS_PER_IDENTITY:]
                update["thumbnails"] = photos
                if not rec.thumbnail:
                    update["thumbnail"] = thumbnail
            updated = rec.model_copy(update=update)
            self._index(updated)
            self._store.save(updated)
            self._bump()
            return True

    def add_photo(self, identity_id: str, photo: bytes) -> bool:
        """Append a user-supplied profile/gallery photo (no recognition template).

        Unlike auto-harvested shots (:meth:`add_embedding`, which carry an
        embedding), this is a curated picture the operator imported — it joins the
        ``thumbnails`` set (bounded, most-recent kept) and becomes the profile
        photo when the identity had none.  Recognition still relies on the
        auto-gathered embeddings; this just lets the user manage how the person
        *looks* in the gallery.
        """
        if not photo:
            return False
        with self._lock:
            rec = self._records.get(identity_id)
            if rec is None:
                return False
            photos = (list(rec.thumbnails) + [photo])[-_MAX_THUMBNAILS_PER_IDENTITY:]
            update: dict[str, object] = {"thumbnails": photos, "updated_at": _now()}
            if not rec.thumbnail:
                update["thumbnail"] = photo
            updated = rec.model_copy(update=update)
            self._index(updated)
            self._store.save(updated)
            self._bump()
            return True

    def set_profile_thumbnail(self, identity_id: str, index: int) -> bool:
        """Choose ``thumbnails[index]`` as the identity's profile photo."""
        with self._lock:
            rec = self._records.get(identity_id)
            if rec is None or not (0 <= index < len(rec.thumbnails)):
                return False
            updated = rec.model_copy(
                update={
                    "thumbnail": rec.thumbnails[index],
                    "updated_at": _now(),
                }
            )
            self._index(updated)
            self._store.save(updated)
            self._bump()
            return True

    def remove_thumbnail(self, identity_id: str, index: int) -> bool:
        """Drop a single captured photo (``thumbnails[index]``) from an identity.

        Lets the user prune a bad / odd-angle recognition shot without deleting
        the whole person.  Only the *thumbnail* is removed — **not** any
        embedding: thumbnails and ``embeddings`` are NOT index-aligned in this
        codebase (they have independent caps and are appended independently by
        :meth:`add_embedding`, whose thumbnail argument is optional), so there is
        no aligned template to drop.  Recognition templates are left intact.

        If the removed photo was the profile ``thumbnail``, the profile falls back
        to the next remaining photo (or ``None`` when none remain).  Persists via
        the store (a labeled identity re-saves its now-shorter photo set; an
        unlabeled one is memory-only by the retention policy).  Never raises;
        returns ``True`` only when a photo was actually removed.
        """
        try:
            with self._lock:
                rec = self._records.get(identity_id)
                if rec is None or not (0 <= index < len(rec.thumbnails)):
                    return False
                photos = list(rec.thumbnails)
                removed = photos.pop(index)
                update: dict[str, object] = {
                    "thumbnails": photos,
                    "updated_at": _now(),
                }
                # Repair the profile photo if it pointed at the removed shot:
                # the next remaining photo takes over (or None when empty).
                if rec.thumbnail == removed:
                    update["thumbnail"] = photos[0] if photos else None
                updated = rec.model_copy(update=update)
                self._index(updated)
                self._store.save(updated)
                self._bump()
                return True
        except Exception:  # noqa: BLE001
            log.warning(
                "remove_thumbnail failed for %s[%s]",
                identity_id,
                index,
                exc_info=True,
            )
            return False

    def label(
        self,
        identity_id: str,
        name: str,
        thumbnail_index: int | None = None,
    ) -> IdentityRecord | None:
        """Promote an unlabeled identity → named + enabled + persisted.

        ``thumbnail_index`` optionally picks the profile photo from the identity's
        candidate ``thumbnails`` at registration time.
        """
        clean = name.strip()
        if not clean:
            return None
        with self._lock:
            rec = self._records.get(identity_id)
            if rec is None:
                return None
            update: dict[str, object] = {
                "name": clean,
                "labeled": True,
                "enabled": True,
                "updated_at": _now(),
            }
            if thumbnail_index is not None and 0 <= thumbnail_index < len(rec.thumbnails):
                update["thumbnail"] = rec.thumbnails[thumbnail_index]
            updated = rec.model_copy(update=update)
            self._index(updated)
            self._store.save(updated)  # now persisted (labeled)
            self._bump()
            return updated

    def rename(self, identity_id: str, new_name: str) -> bool:
        clean = new_name.strip()
        if not clean:
            return False
        with self._lock:
            rec = self._records.get(identity_id)
            if rec is None:
                return False
            updated = rec.model_copy(update={"name": clean, "updated_at": _now()})
            self._index(updated)
            self._store.save(updated)
            self._bump()
            return True

    def set_enabled(self, identity_id: str, enabled: bool) -> bool:
        with self._lock:
            rec = self._records.get(identity_id)
            if rec is None:
                return False
            updated = rec.model_copy(
                update={
                    "enabled": bool(enabled),
                    "updated_at": _now(),
                }
            )
            self._index(updated)
            self._store.save(updated)
            self._bump()
            return True

    def delete(self, identity_id: str) -> bool:
        with self._lock:
            rec = self._records.pop(identity_id, None)
            self._templates.pop(identity_id, None)
            if rec is None:
                return False
            self._store.delete(identity_id)
            self._bump()
            return True

    def expire_unlabeled(self, max_age_s: float) -> list[str]:
        """Drop auto-harvested ("Person N") identities not seen for *max_age_s*.

        Unlabeled records are memory-only scratch identities; left unbounded they
        pile up as a person comes and goes.  Any whose ``updated_at`` (bumped on
        every recognition) is older than the cutoff is removed.  Named (labeled)
        identities are NEVER expired.  Returns the removed ids so the worker can
        tell the UI to forget them.
        """
        if max_age_s <= 0:
            return []
        from datetime import timedelta

        cutoff = _now() - timedelta(seconds=max_age_s)
        removed: list[str] = []
        with self._lock:
            for iid, rec in list(self._records.items()):
                if not rec.labeled and rec.updated_at < cutoff:
                    self._records.pop(iid, None)
                    self._templates.pop(iid, None)
                    removed.append(iid)
            if removed:
                self._bump()
        return removed

    def merge(self, keep_id: str, drop_id: str) -> IdentityRecord | None:
        """Fold ``drop_id``'s embeddings (+ thumbnail) into ``keep_id``.

        The kept identity gains the dropped one's templates (bounded set, most
        recent kept) and inherits its thumbnail only if it had none.  The dropped
        identity is then removed.  If *keep* is labeled it is re-persisted; the
        dropped row is deleted from the store too.
        """
        if keep_id == drop_id:
            return None
        with self._lock:
            keep = self._records.get(keep_id)
            drop = self._records.get(drop_id)
            if keep is None or drop is None:
                return None
            blobs = (list(keep.embeddings) + list(drop.embeddings))[-_MAX_EMBEDDINGS_PER_IDENTITY:]
            thumbnail = keep.thumbnail or drop.thumbnail
            photos = (list(keep.thumbnails) + list(drop.thumbnails))[-_MAX_THUMBNAILS_PER_IDENTITY:]
            merged = keep.model_copy(
                update={
                    "embeddings": blobs,
                    "thumbnail": thumbnail,
                    "thumbnails": photos,
                    "updated_at": _now(),
                }
            )
            self._index(merged)
            self._records.pop(drop_id, None)
            self._templates.pop(drop_id, None)
            self._store.save(merged)
            self._store.delete(drop_id)
            self._bump()
            return merged
