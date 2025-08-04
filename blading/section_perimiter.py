from dataclasses import dataclass
from typing import Optional
from geometry.curves import PlaneCurve


@dataclass
class SectionPerimiter:
    """A class representing the perimeter of a blade section."""

    curve: PlaneCurve
    stream_line: Optional[PlaneCurve]
