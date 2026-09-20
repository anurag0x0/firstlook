from dataclasses import dataclass


@dataclass
class Finding:
    """
    Represents one business-friendly analytical finding.
    """

    importance: int
    category: str
    headline: str
    detail: str
    recommendation: str