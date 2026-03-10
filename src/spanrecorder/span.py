from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid


class SpanStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


def _new_id() -> str:
    # Short, stable string ids (good for logs)
    return uuid.uuid4().hex


def now_ns() -> int:
    # Monotonic clock (safe for durations)
    return time.perf_counter_ns()


@dataclass
class SpanEvent:
    name: str
    ts_ns: int
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A minimal span model for AI/RAG observability.

    - trace_id groups spans for one request/interaction
    - span_id uniquely identifies this span
    - parent_id links to parent span (if any)
    """

    name: str
    trace_id: str
    span_id: str = field(default_factory=_new_id)
    parent_id: Optional[str] = None

    start_ns: int = field(default_factory=now_ns)
    end_ns: Optional[int] = None

    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)

    exception_type: Optional[str] = None
    exception_message: Optional[str] = None

    def end(self, *, status: Optional[SpanStatus] = None, end_ns: Optional[int] = None) -> None:
        """End the span (idempotent)."""
        if self.end_ns is not None:
            return
        self.end_ns = end_ns if end_ns is not None else now_ns()

        if status is not None:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    @property
    def duration_ns(self) -> Optional[int]:
        if self.end_ns is None:
            return None
        return self.end_ns - self.start_ns

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        ts_ns: Optional[int] = None,
    ) -> None:
        self.events.append(
            SpanEvent(
                name=name,
                ts_ns=ts_ns if ts_ns is not None else now_ns(),
                attributes=attributes or {},
            )
        )

    def record_exception(self, exc: BaseException) -> None:
        self.exception_type = type(exc).__name__
        self.exception_message = str(exc)
        self.status = SpanStatus.ERROR

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Span":
        span = cls(
            name=d["name"],
            trace_id=d["trace_id"],
            span_id=d.get("span_id", _new_id()),
            parent_id=d.get("parent_id"),
            start_ns=d.get("start_ns", 0),
            end_ns=d.get("end_ns"),
            status=SpanStatus(d.get("status", "UNSET")),
        )
        for k, v in (d.get("attributes") or {}).items():
            span.set_attribute(k, v)
        for e in d.get("events") or []:
            span.add_event(e["name"], attributes=e.get("attributes"), ts_ns=e.get("ts_ns"))
        exc = d.get("exception")
        if exc and exc.get("type"):
            span.exception_type = exc["type"]
            span.exception_message = exc.get("message")
        return span

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "status": self.status.value,
            "attributes": dict(self.attributes),
            "events": [
                {"name": e.name, "ts_ns": e.ts_ns, "attributes": dict(e.attributes)}
                for e in self.events
            ],
            "exception": (
                {"type": self.exception_type, "message": self.exception_message}
                if self.exception_type is not None
                else None
            ),
        }
