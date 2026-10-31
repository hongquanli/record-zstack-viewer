"""Task 5.1 — the single ndv/OpenGL render surface over a `FovSource`.

Two pieces, and the ONLY module that imports ndv (Global Constraints):

- `FovSourceDataWrapper` adapts the `FovSource` protocol (Task 1.2) to ndv's
  `DataWrapper` base (verified against ndv 0.4.0: abstract `supports`; `dims`/
  `coords` properties; `sizes()`; `isel(index)` where keys are axis names or
  ±ints — normalized via the base's `normalize_axis_key` — and values are ints
  or slices, with slice-valued axes kept as singleton dims because ndv
  transposes/squeezes afterwards). Every requested non-spatial combination is
  materialized with ONE `source.read_plane` call, and requests only cost what
  they select — never a whole-array read on a big source. A raising byte-budget
  guard (`MAX_ISEL_BYTES`) makes that structural: a request that would
  materialize more than the budget (e.g. `isel({})` full-slicing every axis of
  a 10,000-frame recording) errors instead of bulk-reading, while any BOUNDED
  request passes — on a real GL canvas ndv legitimately asks for the channel
  AND z axes wide in one composite z-stack request, which an earlier per-axis
  rule wrongly refused (blank canvas).

- `FovViewer` embeds ONE `ndv.ArrayViewer` and presents the two dataset modes:
  a multi-plane *recording* `(acq_t, frame, z, y, x)` shown grayscale (frame is
  the scrub axis; acq_t steps separately; ndv hides singleton sliders, so the
  z-plane selector appears exactly when n_planes > 1) and a *z-stack*
  `(acq_t, c, z, y, x)` composited across channels. Contrast auto-scales on
  first display from a 1–99.9 percentile of the current decoded plane;
  `omero.window` is a full-dtype range and is used ONLY as clamp bounds.
  Dataset swaps go through ndv's public constructor/setters — an in-place
  `.data`/`.display_model` assignment when the dims signature is unchanged,
  else a clean rebuild of the viewer; no undocumented ndv internals.
"""
from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any, Hashable, Mapping, Sequence

import numpy as np
from cmap import Colormap
from ndv import ArrayViewer, DataWrapper
from ndv.models import ArrayDisplayModel, ClimsManual, LUTModel
from qtpy.QtWidgets import QVBoxLayout, QWidget

from record_zstack_viewer.reader.fov_source import FovSource
from record_zstack_viewer.reader.metadata import ChannelMeta

if TYPE_CHECKING:
    from record_zstack_viewer.reader.adapter import FovDatasets

_AUTO_PERCENTILES = (1.0, 99.9)  # Global Constraints: auto-contrast percentiles
_MODES = ("recording", "zstack")
_SPATIAL = 2  # trailing (y, x) axes of every FovSource
# isel OOM guard: the most bytes one isel() call may materialize. Generous for
# any bounded ndv display request (a multi-channel c×z composite is tens of MB)
# yet far below an accidental whole-array read of a long recording (tens of GB).
MAX_ISEL_BYTES = 256 * 1024 * 1024  # 256 MB


class FovSourceDataWrapper(DataWrapper):
    """ndv `DataWrapper` over a `FovSource` — slices via `read_plane`, never bulk.

    `dims`/`coords` expose the source's NAMED axes (so ndv's sliders and index
    maps are keyed by `acq_t`/`frame`/`c`/`z`), and `isel` maps each requested
    non-spatial combination to one `read_plane` call. ndv's display requests
    are always BOUNDED — width-1 slices for stepped axes plus the visible
    (y, x) axes, and possibly several wide non-spatial axes at once (on a real
    GL canvas a composite z-stack request spans both `c` and `z`) — so they
    stay small regardless of dataset size; the `MAX_ISEL_BYTES` guard in `isel`
    refuses only genuinely unbounded reads (Global Constraints).
    """

    PRIORITY = 40  # checked ahead of ndv's builtin wrappers; `supports` is exact

    _data: FovSource

    def __init__(self, data: FovSource) -> None:
        if not isinstance(data, FovSource):
            raise TypeError(f"FovSourceDataWrapper wraps a FovSource, got {type(data)!r}")
        super().__init__(data)

    @classmethod
    def supports(cls, obj: Any) -> bool:
        """True for anything satisfying the (runtime-checkable) FovSource protocol."""
        return isinstance(obj, FovSource)

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return tuple(self._data.dims)

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        return {d: range(n) for d, n in zip(self._data.dims, self._data.shape)}

    def sizes(self) -> Mapping[Hashable, int]:
        return dict(zip(self._data.dims, self._data.shape))

    def isel(self, index: Mapping[Hashable, int | slice]) -> np.ndarray:
        """Materialize `index` through per-plane reads.

        Keys are axis names or ±ints (normalized via `normalize_axis_key`);
        values are ints (axis dropped, numpy semantics) or slices (axis kept —
        ndv sends width-1 slices for stepped axes and squeezes later). One
        `read_plane` per selected non-spatial combination; the (y, x) selection
        is applied to the returned plane.

        A byte-budget guard enforces the bounded-memory Global Constraint:
        if the selected widths (over ALL axes) would materialize more than
        `MAX_ISEL_BYTES`, raise ValueError BEFORE reading anything — that size
        only arises from an effectively whole-array read (e.g. `isel({})` on a
        long recording), never from an ndv display request. Any bounded
        request passes, including composite z-stack requests that span both
        the `c` and `z` axes (raising, not truncating: a narrowed result
        would silently break ndv's shape contract).
        """
        shape, dims = self._data.shape, self._data.dims
        sel: list[int | slice] = [slice(None)] * len(shape)
        for key, val in index.items():
            sel[self.normalize_axis_key(key)] = val
        ns = len(shape) - _SPATIAL

        picks: list[range] = []  # concrete indices to fetch, per non-spatial axis
        kept: list[bool] = []    # slice keeps the axis; an int drops it
        for ax in range(ns):
            val = sel[ax]
            if isinstance(val, slice):
                picks.append(range(*val.indices(shape[ax])))
                kept.append(True)
            else:
                i = int(val)
                if i < 0:
                    i += shape[ax]
                picks.append(range(i, i + 1))
                kept.append(False)

        yx = tuple(sel[ns:])
        spatial_widths = [
            len(range(*v.indices(shape[ns + j]))) if isinstance(v, slice) else 1
            for j, v in enumerate(yx)]
        n_planes = math.prod(len(r) for r in picks)
        n_bytes = n_planes * math.prod(spatial_widths) * self.dtype.itemsize
        if n_bytes > MAX_ISEL_BYTES:
            raise ValueError(
                f"isel index {dict(index)!r} exceeds the plane budget: it "
                f"would materialize {n_bytes:,} bytes ({n_planes} planes), "
                f"over MAX_ISEL_BYTES = {MAX_ISEL_BYTES:,}. Refusing what is "
                "effectively a whole-array read — ndv display requests are "
                "bounded and stay far under the budget.")

        spatial_shape = tuple(
            w for j, w in enumerate(spatial_widths) if isinstance(yx[j], slice))
        out_shape = tuple(len(picks[ax]) for ax in range(ns) if kept[ax]) + spatial_shape
        out = np.empty(out_shape, dtype=self.dtype)
        for pos in itertools.product(*(range(len(r)) for r in picks)):
            plane_index = {dims[ax]: picks[ax][pos[ax]] for ax in range(ns)}
            plane = self._data.read_plane(plane_index)  # ONE visible plane
            out[tuple(p for ax, p in enumerate(pos) if kept[ax])] = plane[yx]
        return out


# ------------------------- auto-contrast helpers ----------------------------

def _auto_clims(plane: np.ndarray, window: tuple[float, float]) -> tuple[float, float]:
    """1–99.9 percentile limits of `plane`, clamped into `window`.

    `window` is the omero channel window — a full-dtype clamp range, NOT a
    display default (Global Constraints) — so it only bounds the result. A
    constant plane (or a collapsed clamp) is widened minimally so the LUT
    range never degenerates.
    """
    p_lo, p_hi = np.percentile(plane, _AUTO_PERCENTILES)
    w_lo, w_hi = float(window[0]), float(window[1])
    lo = min(max(float(p_lo), w_lo), w_hi)
    hi = max(min(float(p_hi), w_hi), w_lo)
    if not hi > lo:
        hi = lo + 1.0
    return lo, hi


def _dtype_window(dtype: np.dtype) -> tuple[float, float]:
    """Fallback clamp bounds when a channel has no omero window: the dtype range."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def _channel_meta(source: FovSource, ci: int) -> ChannelMeta:
    """`metadata.channels[ci]`, or a neutral white channel when absent."""
    channels = source.metadata.channels
    if ci < len(channels):
        return channels[ci]
    return ChannelMeta(label=f"ch{ci}", color="#FFFFFF",
                       window=_dtype_window(source.dtype), active=True)


def _colormap(color: str) -> Colormap:
    """Black→`color` ramp (the microscopy channel LUT); gray if unparsable."""
    try:
        return Colormap(["#000000", color])
    except Exception:
        return Colormap("gray")


def _auto_lut(source: FovSource, plane_index: dict[str, int], ch: ChannelMeta) -> LUTModel:
    """A LUTModel whose limits come from the current decoded plane's percentiles."""
    lo, hi = _auto_clims(source.read_plane(plane_index), ch.window)
    return LUTModel(cmap=_colormap(ch.color), clims=ClimsManual(min=lo, max=hi),
                    clim_bounds=(ch.window[0], ch.window[1]), visible=ch.active)


def _display_model_for(source: FovSource, mode: str) -> ArrayDisplayModel:
    """The initial ndv display model for `source` in `mode`, LUTs pre-contrasted.

    `current_index` is seeded with NAMED keys covering every stepped axis —
    ndv's sliders are keyed by our named coords, so the first data request is
    already fully specified (one visible plane; the C channel planes in
    composite mode) and never falls back to whole-axis slices.
    """
    nz = source.shape[source.dims.index("z")]
    current: dict[str, int] = {"acq_t": 0, "z": nz // 2}
    if mode == "zstack":
        n_ch = source.shape[source.dims.index("c")]
        luts = {
            ci: _auto_lut(source, {"acq_t": 0, "c": ci, "z": nz // 2},
                          _channel_meta(source, ci))
            for ci in range(n_ch)}
        return ArrayDisplayModel(visible_axes=("y", "x"), channel_axis="c",
                                 channel_mode="composite", current_index=current,
                                 luts=luts)
    current["frame"] = 0
    lut = _auto_lut(source, dict(current), _channel_meta(source, 0))
    return ArrayDisplayModel(visible_axes=("y", "x"), channel_axis=None,
                             channel_mode="grayscale", current_index=current,
                             default_lut=lut)


# --------------------------------- widget -----------------------------------

class FovViewer(QWidget):
    """One FOV's pixels in a single embedded `ndv.ArrayViewer`.

    `set_datasets(datasets, mode)` selects the acquisition phase ("recording" |
    "zstack") and shows it with auto-contrasted LUTs; `set_timepoint` steps the
    shared `acq_t` axis; `auto_contrast()` re-derives the percentile limits
    from the currently displayed plane(s). Swaps reuse the live viewer via the
    public `.data`/`.display_model` setters when the dims signature matches,
    and otherwise rebuild the ArrayViewer through its public constructor.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._viewer: ArrayViewer | None = None
        self._source: FovSource | None = None
        self._mode: str | None = None
        self._recording_lut: LUTModel | None = None  # the LUT the last recording was shown with

    @property
    def mode(self) -> str | None:
        """The dataset mode on display ("recording" | "zstack"), or None."""
        return self._mode

    @property
    def ndv_viewer(self) -> ArrayViewer | None:
        """The embedded ndv ArrayViewer (opaque to callers that don't import ndv)."""
        return self._viewer

    def current_lut(self) -> tuple[tuple[float, float] | None, str | None]:
        """What the displayed recording is painted with, for the mp4 export to
        copy: `((lo, hi), "#RRGGBB")`, the manual contrast limits and the top
        colour of the channel ramp. `(None, None)` when nothing or a z-stack is
        shown (only recordings are exported)."""
        if self._viewer is None or self._mode != "recording":
            return None, None
        lut = self._viewer.display_model.default_lut
        clims = lut.clims
        window = (float(clims.min), float(clims.max)) if isinstance(clims, ClimsManual) else None
        r, g, b = (int(round(float(v) * 255)) for v in lut.cmap(1.0)[:3])
        return window, f"#{r:02X}{g:02X}{b:02X}"

    def set_datasets(self, datasets: FovDatasets, mode: str) -> None:
        """Show `datasets`' phase `mode` ("recording" | "zstack")."""
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        source: FovSource | None = getattr(datasets, mode)
        if source is None:
            raise ValueError(f"datasets has no {mode!r} phase for this FOV")
        self.set_source(source, mode)

    def set_source(self, source: FovSource, mode: str) -> None:
        """Show `source` directly (same contract as `set_datasets`)."""
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        wrapper = FovSourceDataWrapper(source)
        model = _display_model_for(source, mode)
        self._carry_recording_lut(model, mode)
        if (self._viewer is not None and self._source is not None
                and tuple(source.dims) == tuple(self._source.dims)):
            try:
                # public setters; each re-synchronizes the view. Data first: the
                # old display model shares the dims signature, so the transient
                # sync stays valid, then the fresh model (auto-contrast LUTs,
                # reset index) replaces it.
                self._viewer.data = wrapper
                self._viewer.display_model = model
            except Exception:  # pragma: no cover - defensive: never rely on in-place
                self._rebuild(wrapper, model)
        else:
            self._rebuild(wrapper, model)
        self._source, self._mode = source, mode

    def _carry_recording_lut(self, model: ArrayDisplayModel, mode: str) -> None:
        """Recordings keep the contrast limits and colormap across FOV switches:
        the user sets them once and every well, and the mp4 export, uses them.
        Only the first recording shown auto-contrasts (`auto_contrast()` resets
        on demand). The LUT is remembered when leaving recording mode, so a
        detour through a z-stack does not lose it."""
        if self._viewer is not None and self._mode == "recording":
            self._recording_lut = self._viewer.display_model.default_lut
        kept = self._recording_lut
        if mode == "recording" and kept is not None and isinstance(kept.clims, ClimsManual):
            model.default_lut.clims = ClimsManual(min=kept.clims.min, max=kept.clims.max)
            model.default_lut.cmap = kept.cmap

    def _rebuild(self, wrapper: FovSourceDataWrapper, model: ArrayDisplayModel) -> None:
        """Clean fallback: a fresh ArrayViewer via its public constructor."""
        old, self._viewer = self._viewer, ArrayViewer(wrapper, display_model=model)
        if old is not None:
            old_widget = old.widget()
            self._layout.removeWidget(old_widget)
            old_widget.setParent(None)
            old_widget.deleteLater()
            old.close()
        self._layout.addWidget(self._viewer.widget())

    def update_extent(self, source: FovSource) -> None:
        """Data-only live swap: show `source` (a re-opened FOV whose store grew)
        while KEEPING the current display model.

        The user's slider index and contrast limits survive — unlike
        `set_datasets`, which installs a fresh model (auto-contrast, reset
        index). ndv's public ``.data`` setter re-reads the wrapper's dims/coords
        (so a grown axis extends its slider) and re-requests the current slice,
        surfacing the new pixels. The dims signature must match the displayed
        source (same phase); anything else raises so the caller can fall back
        to a full `set_datasets` rebuild.
        """
        if self._viewer is None or self._source is None:
            raise RuntimeError("set_datasets() before update_extent()")
        if tuple(source.dims) != tuple(self._source.dims):
            raise ValueError(
                f"update_extent needs the displayed dims {tuple(self._source.dims)}, "
                f"got {tuple(source.dims)}; use set_datasets for a phase change")
        self._viewer.data = FovSourceDataWrapper(source)  # display model untouched
        self._source = source

    def set_timepoint(self, t: int) -> None:
        """Step the shared `acq_t` axis to `t` (clamped to the source's extent)."""
        if self._viewer is None or self._source is None:
            raise RuntimeError("set_datasets() before set_timepoint()")
        n_t = self._source.shape[0]  # acq_t is axis 0 in both modes
        self._viewer.display_model.current_index.update(
            {"acq_t": max(0, min(int(t), n_t - 1))})

    @property
    def timepoint(self) -> int:
        """The current shared `acq_t` index (0 when nothing is displayed)."""
        if self._viewer is None:
            return 0
        t = self._viewer.display_model.current_index.get("acq_t", 0)
        return int(t) if isinstance(t, int) else 0

    def auto_contrast(self) -> None:
        """Recompute the 1–99.9 percentile limits from the current plane(s)."""
        if self._viewer is None or self._source is None:
            raise RuntimeError("set_datasets() before auto_contrast()")
        source, dm = self._source, self._viewer.display_model
        cur = {k: v for k, v in dm.current_index.items() if isinstance(v, int)}
        nz = source.shape[source.dims.index("z")]
        base = {"acq_t": int(cur.get("acq_t", 0)), "z": int(cur.get("z", nz // 2))}
        if self._mode == "zstack":
            n_ch = source.shape[source.dims.index("c")]
            for ci in range(n_ch):
                ch = _channel_meta(source, ci)
                lo, hi = _auto_clims(source.read_plane(base | {"c": ci}), ch.window)
                if (lut := dm.luts.get(ci)) is not None:
                    lut.clims = ClimsManual(min=lo, max=hi)
                else:  # a channel ndv hasn't materialized yet
                    dm.luts[ci] = _auto_lut(source, base | {"c": ci}, ch)
        else:
            ch = _channel_meta(source, 0)
            plane = source.read_plane(base | {"frame": int(cur.get("frame", 0))})
            lo, hi = _auto_clims(plane, ch.window)
            dm.default_lut.clims = ClimsManual(min=lo, max=hi)
