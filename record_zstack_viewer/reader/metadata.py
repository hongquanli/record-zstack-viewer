"""Per-store metadata parsing (CONTRACT.md): `ome.omero` channels, `_squid`
scalars/flags, the per-FOV placement `translation`, and the yaml sidecars.

Tolerant by design (contract rule: always have a fallback) — missing keys yield
neutral defaults, never exceptions.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

_FULL_UINT16 = (0.0, 65535.0)  # omero.window is a full-dtype clamp range, not a display default


@dataclass
class ChannelMeta:
    label: str
    color: str  # hex, passed through as written by the producer
    window: tuple[float, float]  # (start, end) clamp bounds
    active: bool


@dataclass
class FovMetadata:
    pixel_size_um: float
    z_step_um: float
    time_increment_s: float
    channels: list[ChannelMeta]
    acquisition_complete: bool
    aborted: bool
    n_planes: int  # sibling plane stores of a multi-plane recording; absence => 1
    # per-plane z offsets are collected across sibling stores by the caller (Task 1.4)
    plane_z_offsets: list[float] | None = None
    # (x, y, z) stage placement in um from the OME `translation`; None when absent (old data)
    position_um: tuple[float, float, float] | None = None


def read_store_zarr_json(store_dir: Path) -> dict:
    """Parse `<store>/zarr.json` (recording: array node at root; z-stack: group node)."""
    return json.loads((Path(store_dir) / "zarr.json").read_text())


def _transforms(attrs: dict) -> dict[str, list]:
    """{type: values} from ome.multiscales[0].datasets[0].coordinateTransformations."""
    try:
        xforms = attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    except (KeyError, IndexError, TypeError):
        return {}
    return {x["type"]: x[x["type"]] for x in xforms
            if isinstance(x, dict) and x.get("type") in ("scale", "translation")
            and isinstance(x.get(x["type"]), (list, tuple))}


def _channel(ch: dict) -> ChannelMeta:
    w = ch.get("window") or {}
    return ChannelMeta(label=str(ch.get("label", "")),
                       color=str(ch.get("color", "#FFFFFF")),
                       window=(float(w.get("start", _FULL_UINT16[0])),
                               float(w.get("end", _FULL_UINT16[1]))),
                       active=bool(ch.get("active", True)))


def fov_metadata_from_attrs(attrs: dict) -> FovMetadata:
    """Build FovMetadata from a store's `zarr.json` `attributes` dict."""
    squid = attrs.get("_squid") or {}
    tf = _transforms(attrs)
    scale = tf.get("scale") or []

    def scalar(key: str, scale_idx: int) -> float:
        # prefer _squid; fall back to the OME scale transform (t,c,z,y,x); neutral 1.0 last
        v = squid.get(key)
        if v is None and len(scale) == 5:
            v = scale[scale_idx]
        return float(v) if v is not None else 1.0

    translation = tf.get("translation")
    position = ((float(translation[4]), float(translation[3]), float(translation[2]))
                if translation is not None and len(translation) == 5 else None)  # (x, y, z) um
    omero = (attrs.get("ome") or {}).get("omero") or {}
    return FovMetadata(
        pixel_size_um=scalar("pixel_size_um", 4),
        z_step_um=scalar("z_step_um", 2),
        time_increment_s=scalar("time_increment_s", 0),
        channels=[_channel(ch) for ch in omero.get("channels") or []],
        acquisition_complete=bool(squid.get("acquisition_complete", False)),
        aborted=bool(squid.get("aborted", False)),
        n_planes=int(squid.get("n_planes", 1)),
        position_um=position,
    )


def parse_acquisition_yaml(path: Path) -> dict:
    """acquisition.yaml (written at run START): wellplate_scan geometry, xy_mode, ..."""
    return yaml.safe_load(Path(path).read_text()) or {}


def parse_channels_yaml(path: Path) -> list[ChannelMeta]:
    """acquisition_channels.yaml (run START): channel names (+ colors when present)."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    out = []
    for ch in doc.get("channels") or []:
        if isinstance(ch, str):
            ch = {"name": ch}
        out.append(ChannelMeta(label=str(ch.get("name") or ch.get("label") or ""),
                               color=str(ch.get("color", "#FFFFFF")),
                               window=_FULL_UINT16, active=True))
    return out
