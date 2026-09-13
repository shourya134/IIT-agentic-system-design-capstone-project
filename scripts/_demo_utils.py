"""Shared helper for demo scripts: capture printed lines and save them as the
task's transcript file under transcripts/."""
from pathlib import Path

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "transcripts"


class Logger:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, *args) -> None:
        text = " ".join(str(a) for a in args)
        print(text)
        self.lines.append(text)

    def save(self, filename: str) -> None:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        (TRANSCRIPTS_DIR / filename).write_text("\n".join(self.lines) + "\n", encoding="utf-8")
