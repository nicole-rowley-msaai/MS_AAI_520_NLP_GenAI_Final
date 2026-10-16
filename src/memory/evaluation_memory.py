"""Independent reflection memory for the Evaluation Lead branch.

Stores only validated lesson records in a separate JSON file. Retrieval
prioritizes ticker, sector, and general notes, then recency within each
category. This does not retrain the LLM and is not integrated into graph.py.

The checks are structural heuristics, not proof that a natural language
lesson is actionable or that a financial claim is correct. AI assisted
implementation must be disclosed under the course requirements.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.agents.schemas import Draft, Finding, Reflection, ToolName

AVAILABLE_TOOLS = frozenset(ToolName.__args__)
TOOL_REFERENCE = re.compile(r"\b(?:get|search)_[a-zA-Z0-9_]+\b", re.I)
WHITESPACE = re.compile(r"\s+")
MAX_RECORDS = 60


def normalize_lesson(value: str) -> str:
    """Normalize repeated whitespace for reliable duplicate comparison."""
    return WHITESPACE.sub(" ", value).strip()


def verify_lesson(text: str) -> str | None:
    """Return a rejection reason, or None if basic checks pass.

    Tool checks apply ONLY to explicit function names. This function does
    not assess whether free form advice can be executed by the planner.
    """
    if not isinstance(text, str):
        return "not_text"
    cleaned = normalize_lesson(text)
    if not cleaned:
        return "empty"
    if len(cleaned) < 12:
        return "too_short"
    if len(cleaned) > 240:
        return "too_long"
    mentioned = {name.lower() for name in TOOL_REFERENCE.findall(cleaned)}
    if mentioned - AVAILABLE_TOOLS:
        return "unknown_explicit_tool"
    return None


def note_priority(note: dict, ticker: str, sector: str | None) -> int:
    """0 ticker, 1 sector, 2 general, 3 irrelevant."""
    wanted_ticker = ticker.strip().casefold()
    noted_ticker = str(note.get("ticker") or "").strip().casefold()
    wanted_sector = (sector or "").strip().casefold()
    noted_sector = str(note.get("sector") or "").strip().casefold()
    if noted_ticker == wanted_ticker and wanted_ticker:
        return 0
    if wanted_sector and noted_sector == wanted_sector:
        return 1
    if noted_ticker in ("", "*"):
        return 2
    return 3


def sort_date(note: dict) -> date:
    """Malformed dates are treated as the oldest possible, not newest."""
    try:
        return date.fromisoformat(str(note.get("date", "")))
    except ValueError:
        return date.min


def select_lessons(
    notes: Iterable[dict], ticker: str, sector: str | None, limit: int = 8
) -> list[dict]:
    """Select relevant records, newest first within each relevance rank."""
    if limit < 0:
        raise ValueError("limit cannot be negative")
    ranked = [
        (note_priority(n, ticker, sector), n)
        for n in notes
        if isinstance(n, dict) and isinstance(n.get("lesson"), str)
    ]
    ranked = [(priority, n) for priority, n in ranked if priority < 3]
    # Stable two pass ordering: most recent first, then highest priority.
    ranked.sort(key=lambda row: sort_date(row[1]), reverse=True)
    ranked.sort(key=lambda row: row[0])
    return [dict(n) for _, n in ranked[:limit]]


@dataclass(frozen=True)
class LessonRejection:
    lesson: str
    reason: str


@dataclass(frozen=True)
class MemoryResult:
    reflection: Reflection
    previous_notes: tuple[dict, ...]
    added_notes: tuple[dict, ...]
    rejected: tuple[LessonRejection, ...]
    saved: bool


@dataclass(frozen=True)
class EvaluationMemory:
    """Explicit-path storage so unit tests never modify team memory."""

    path: Path
    max_records: int = MAX_RECORDS

    def __post_init__(self) -> None:
        if self.max_records < 1:
            raise ValueError("max_records must be positive")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("memory file could not be read as JSON") from exc
        if not isinstance(records, list) or not all(
            isinstance(record, dict) and isinstance(record.get("lesson"), str)
            for record in records
        ):
            raise ValueError(
                "memory file must contain a list of lesson records"
            )
        return records

    def relevant(self, ticker: str, sector: str | None, limit: int = 8):
        return select_lessons(self.load(), ticker, sector, limit=limit)

    def _write(self, notes: list[dict]) -> None:
        """Replace the target file after successful serialization."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(notes[-self.max_records :], indent=2) + "\n"
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".evaluation_notes_",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_name = handle.name
                handle.write(payload)
            os.replace(temp_name, self.path)
        finally:
            if temp_name is not None and os.path.exists(temp_name):
                os.unlink(temp_name)

    def add_reflection(
        self,
        ticker: str,
        sector: str | None,
        reflection: Reflection,
        *,
        today: date | None = None,
        persist: bool = False,
    ) -> MemoryResult:
        """Screen reflection lessons and optionally persist new records.

        The default is preview only. Use persist=True explicitly to write.
        """
        previous = self.load()
        existing = {
            normalize_lesson(n["lesson"]).casefold() for n in previous
        }
        new = []
        rejected = []
        run_date = (today or date.today()).isoformat()
        for raw in reflection.lessons:
            reason = verify_lesson(raw)
            cleaned = normalize_lesson(raw) if isinstance(raw, str) else ""
            if reason:
                rejected.append(LessonRejection(str(raw), reason))
                continue
            key = cleaned.casefold()
            if key in existing:
                rejected.append(LessonRejection(cleaned, "duplicate"))
                continue
            existing.add(key)
            new.append(
                {
                    "ticker": ticker.strip().upper(),
                    "sector": (sector or "").strip(),
                    "date": run_date,
                    "lesson": cleaned,
                }
            )
        if persist and new:
            self._write(previous + new)
        return MemoryResult(
            reflection=reflection,
            previous_notes=tuple(dict(n) for n in previous),
            added_notes=tuple(dict(n) for n in new),
            rejected=tuple(rejected),
            saved=bool(persist and new),
        )


def reflect_and_remember(
    draft: Draft,
    findings: list[Finding],
    tool_log: list[dict],
    *,
    ticker: str,
    sector: str | None,
    memory: EvaluationMemory,
    reflect_fn: Callable[[Draft, list[Finding], list[dict]], Reflection],
    today: date | None = None,
    persist: bool = False,
) -> MemoryResult:
    """Run an injected reflection function and screen lessons for storage.

    Callers may provide ev.reflect for a live LLM run or a deterministic test
    callback. No LLM is called by this module without that explicit callback.
    """
    if draft.ticker.strip().upper() != ticker.strip().upper():
        raise ValueError("reflection ticker does not match the draft")
    reflection = reflect_fn(draft, findings, tool_log)
    return memory.add_reflection(
        ticker, sector, reflection, today=today, persist=persist
    )
