"""Pure reconnect-policy helper for PTZ transports.

No I/O — only time-based backoff bookkeeping.  Both ViscaIPBackend and
ViscaUSBBackend hold one instance and consult it after every transport error.
"""

from __future__ import annotations


class ReconnectPolicy:
    """Exponential-backoff reconnect gate.

    Args:
        base_s: Base backoff delay in seconds (first failure waits this long).
        cap_s:  Maximum backoff delay; subsequent failures are capped at this.

    Usage::

        policy = ReconnectPolicy()
        ...
        except OSError:
            policy.record_failure(now)
            if policy.should_attempt(now):
                _open()
                policy.record_success()
    """

    def __init__(self, base_s: float = 1.0, cap_s: float = 30.0) -> None:
        self._base_s = base_s
        self._cap_s = cap_s
        self._attempts: int = 0
        self._next_allowed_t: float = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def should_attempt(self, now: float) -> bool:
        """Return True when a reconnect attempt is permitted right now."""
        return now >= self._next_allowed_t

    def record_failure(self, now: float) -> None:
        """Record a transport failure and schedule the next permitted attempt."""
        self._attempts += 1
        delay = min(self._cap_s, self._base_s * (2 ** (self._attempts - 1)))
        self._next_allowed_t = now + delay

    def record_success(self) -> None:
        """Reset the policy after a successful (re)connection."""
        self._attempts = 0
        self._next_allowed_t = 0.0
