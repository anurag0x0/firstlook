from dataclasses import dataclass


@dataclass
class Issue:
    """Represents a single data-quality issue."""

    severity: str
    kind: str
    column: str | None
    message: str
    suggestion: str
    detail: dict