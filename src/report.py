"""Plain-text run report writer (``run_info.txt``)."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _row(k: str, v: Any, pad: int = 13) -> str:
    return f"{k.ljust(pad)}: {v}"


@dataclass
class RunReport:
    out_dir: Path
    model: str
    quant: str = "none"
    sage_attn: bool = False
    args: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = self.out_dir / "run_info.txt"
        lines = []
        lines.append("[run]")
        lines.append(_row("date", _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        lines.append(_row("model", self.model))
        lines.append(_row("quant", self.quant))
        lines.append(_row("sage_attn", "enabled" if self.sage_attn else "disabled"))
        lines.append("")
        lines.append("[args]")
        for k, v in self.args.items():
            lines.append(_row(k, v))
        lines.append("")
        lines.append("[timing]")
        for k, v in self.timing.items():
            lines.append(_row(k, v))
        if self.extra:
            lines.append("")
            lines.append("[extra]")
            for k, v in self.extra.items():
                lines.append(_row(k, v))
        text = "\n".join(lines) + "\n"
        out.write_text(text, encoding="utf-8")
        return out
