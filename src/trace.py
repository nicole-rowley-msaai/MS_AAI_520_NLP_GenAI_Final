"""RunTrace: a record of every step the agent takes in one run.

Each graph node appends an event. The notebook renders the trace so a
reader can see the plan, every tool call, every routing decision, each
evaluator iteration, the reflection, and the memory change without
opening the code. Runs are saved to runs/<ticker>_<timestamp>.json.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import ROOT

RUNS_DIR = ROOT / "runs"


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion so any tool output can be saved."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class Event:
    step: str        # e.g. "planner", "tool", "router", "evaluator"
    kind: str        # e.g. "plan", "call", "decision", "scores"
    input: Any = None
    output: Any = None
    seconds: float = 0.0


@dataclass
class RunTrace:
    ticker: str
    started: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    events: list[Event] = field(default_factory=list)

    def log(self, step: str, kind: str, input: Any = None,
            output: Any = None, seconds: float = 0.0) -> None:
        self.events.append(Event(
            step, kind, _jsonable(input), _jsonable(output), round(seconds, 2),
        ))

    def timed(self, step: str, kind: str, func, *args, **kwargs):
        """Run func, log its input and output with elapsed time, return output."""
        start = time.perf_counter()
        out = func(*args, **kwargs)
        self.log(step, kind, {"args": args, **kwargs}, out,
                 time.perf_counter() - start)
        return out

    def select(self, step: str | None = None, kind: str | None = None) -> list[Event]:
        return [e for e in self.events
                if (step is None or e.step == step)
                and (kind is None or e.kind == kind)]

    def save(self, path: Path | None = None) -> Path:
        RUNS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = path or RUNS_DIR / f"{self.ticker}_{stamp}.json"
        path.write_text(json.dumps(asdict(self), indent=1, default=str))
        return path

    @classmethod
    def load(cls, path: Path) -> "RunTrace":
        data = json.loads(Path(path).read_text())
        events = [Event(**e) for e in data.pop("events")]
        return cls(events=events, **data)
