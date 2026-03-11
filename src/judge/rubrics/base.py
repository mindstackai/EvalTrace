from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol


@dataclass(frozen=True, slots=True)
class RubricScore:
    key: str
    scale_min: float = 0.0
    scale_max: float = 1.0
    description: str = ""


class Rubric(Protocol):
    name: str
    instructions: str
    dimensions: List[RubricScore]

    def as_dict(self) -> Dict: ...
