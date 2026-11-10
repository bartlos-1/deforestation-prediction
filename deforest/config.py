from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path = ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        p = p if p.is_absolute() else self.root / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def bbox(self) -> list[float]:
        return self.raw["region"]["bbox"]


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for k in parents:
            node = node[k]
        node[leaf] = value
    # relative paths resolve against the directory holding the config file
    return Config(raw=raw, root=path.resolve().parent)
