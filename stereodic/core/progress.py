"""Progress reporting and cooperative cancellation for long-running stages.

Pipeline stages accept an optional ``progress`` object. They call ``update`` at
natural boundaries (calibration view, DIC frame, reconstruction frame), may send
small per-frame ``summary`` records, and call ``check_cancel`` only where the
results file is consistent, so a cancelled stage can be resumed.

The default :data:`NULL_PROGRESS` does nothing, so command-line use is unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class JobCancelled(Exception):
    """Raised by :meth:`ProgressReporter.check_cancel` when a cancel was requested."""


@runtime_checkable
class ProgressReporter(Protocol):
    def update(self, done: int, total: int, message: str = "") -> None:
        """``done`` of ``total`` work units of the current stage are finished."""

    def phase(self, name: str) -> None:
        """A named step inside the stage started (e.g. "matching", "writing")."""

    def summary(self, data: dict[str, Any]) -> None:
        """Small JSON-serialisable record, e.g. per-frame valid ratios."""

    def check_cancel(self) -> None:
        """Raise :class:`JobCancelled` if a cancel was requested."""


class NullProgress:
    """Reporter that ignores everything and never cancels."""

    def update(self, done: int, total: int, message: str = "") -> None:
        pass

    def phase(self, name: str) -> None:
        pass

    def summary(self, data: dict[str, Any]) -> None:
        pass

    def check_cancel(self) -> None:
        pass


NULL_PROGRESS = NullProgress()


def ensure_progress(progress: ProgressReporter | None) -> ProgressReporter:
    return NULL_PROGRESS if progress is None else progress
