"""Cross-run memory: short lessons the planner reads before planning.

memory/notes.json is a list of records:
    {"ticker": "JPM", "sector": "Financial Services",
     "date": "2026-10-09", "lesson": "For banks, pull net interest margin."}

It is deliberately plain JSON so the notebook can print a before/after diff.
"""

import json
from datetime import date
from pathlib import Path

from src.config import ROOT

NOTES_PATH = ROOT / "memory" / "notes.json"
MAX_NOTES = 60  # keep the file short; the planner reads the relevant subset


def load_notes(path: Path = NOTES_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_notes(notes: list[dict], path: Path = NOTES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notes[-MAX_NOTES:], indent=1))


def relevant_notes(
    ticker: str, sector: str | None, notes: list[dict], limit: int = 8
) -> list[dict]:
    """Same ticker first, then same sector, then general lessons."""

    def rank(n: dict) -> int:
        if n.get("ticker") == ticker:
            return 0
        if sector and n.get("sector") == sector:
            return 1
        if n.get("ticker") in (None, "", "*"):
            return 2
        return 3

    ranked = sorted(notes, key=lambda n: (rank(n), n.get("date", "")))
    return [n for n in ranked if rank(n) < 3][:limit]


def add_lessons(
    ticker: str,
    sector: str | None,
    lessons: list[str],
    path: Path = NOTES_PATH,
) -> list[dict]:
    """Append this run's lessons; returns the new records."""
    notes = load_notes(path)
    today = date.today().isoformat()
    existing = {n["lesson"] for n in notes}
    new = [
        {
            "ticker": ticker,
            "sector": sector or "",
            "date": today,
            "lesson": text,
        }
        for text in lessons
        if text and text not in existing
    ]
    save_notes(notes + new, path)
    return new


def format_notes(notes: list[dict]) -> str:
    if not notes:
        return "(no prior lessons)"
    return "\n".join(
        f"- [{n.get('ticker') or 'general'}, {n.get('date')}] {n['lesson']}"
        for n in notes
    )
