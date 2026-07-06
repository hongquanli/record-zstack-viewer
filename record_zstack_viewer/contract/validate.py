"""Structural validator for the record-zstack data contract (see CONTRACT.md).

Errors = stable-contract violations; warnings = missing optional sidecars.
"""
from dataclasses import dataclass, field
from pathlib import Path

import tensorstore as ts


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _open(path: Path):
    return ts.open({"driver": "zarr3",
                    "kvstore": {"driver": "file", "path": str(path)}}).result()


def validate_experiment(root: Path) -> ValidationReport:
    root = Path(root)
    r = ValidationReport()
    if not (root / "acquisition.yaml").exists():
        r.errors.append("missing acquisition.yaml")
    if not (root / "acquisition_channels.yaml").exists():
        r.errors.append("missing acquisition_channels.yaml")
    for store in sorted((root / "recording").rglob("*.ome.zarr")):
        try:  # recording: array lives at the store root
            a = _open(store)
            if a.ndim != 5:
                r.errors.append(f"{store}: ndim {a.ndim} != 5")
        except Exception as e:
            r.errors.append(f"{store}: {e}")
    for grp in sorted((root / "zarr").rglob("*.ome.zarr")):
        try:  # z-stack: array lives at <store>/0
            a = _open(grp / "0")
            if a.ndim != 5:
                r.errors.append(f"{grp}/0: ndim {a.ndim} != 5")
        except Exception as e:
            r.errors.append(f"{grp}/0: {e}")
    for opt in ("coordinates.csv", "events.jsonl", "thumbnails"):
        if not (root / opt).exists():
            r.warnings.append(f"optional {opt} absent (fallback used)")
    r.ok = not r.errors
    return r
