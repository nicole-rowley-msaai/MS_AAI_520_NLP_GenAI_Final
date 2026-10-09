"""Data tools the agent can call. Each returns plain data or a ToolResult."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Uniform return type so one failed API never crashes a run."""

    tool: str
    ok: bool
    data: Any = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


def safe_call(tool_name: str, fn, *args, **kwargs) -> ToolResult:
    """Run a tool and wrap the outcome; errors become ok=False results."""
    try:
        data = fn(*args, **kwargs)
        rows = len(data) if hasattr(data, "__len__") else None
        return ToolResult(
            tool=tool_name, ok=True, data=data, meta={"rows": rows}
        )
    except Exception as exc:  # noqa: BLE001 - report any tool failure
        return ToolResult(
            tool=tool_name, ok=False, error=f"{type(exc).__name__}: {exc}"
        )
