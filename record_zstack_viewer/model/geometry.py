"""Pure plate/scan geometry: well-name math, plate-format table, FOV-grid
reconstruction, and scan-pattern index maps.

No Qt, no IO, no tensorstore — every symbol here is a pure function or a frozen
constant. This module is the **fallback** for per-FOV placement: the primary
path derives a FOV's position from the OME `translation`
(`FovMetadata.position_um`), and this geometry only reconstructs positions for
pre-translation data. The scan-pattern order therefore has to be exactly right
for that older data, which is why `spattern_index_to_ij` is nailed down by test.

Plate-format numbers are copied from squid's `sample_formats.csv`.
"""
import math
from dataclasses import dataclass

__all__ = [
    "PlateFormat",
    "PLATE_FORMATS",
    "well_name_to_rc",
    "infer_plate_format",
    "fov_grid_shape",
    "spattern_index_to_ij",
    "unidirectional_index_to_ij",
]


@dataclass(frozen=True)
class PlateFormat:
    """Physical layout of a well plate (squid `sample_formats.csv` values)."""

    rows: int
    cols: int
    well_spacing_mm: float          # center-to-center pitch, same in x and y
    a1_mm: tuple[float, float]      # (x, y) stage coords of the A1 well center
    well_size_mm: float             # well diameter / edge length


# name -> layout. Values from squid `sample_formats.csv` (a1 = (x, y) mm).
PLATE_FORMATS: dict[str, PlateFormat] = {
    "6": PlateFormat(2, 3, 39.2, (24.55, 23.01), 34.94),
    "12": PlateFormat(3, 4, 26.0, (24.75, 16.86), 22.05),
    "24": PlateFormat(4, 6, 19.3, (24.45, 22.07), 15.54),
    "96": PlateFormat(8, 12, 9.0, (11.31, 10.75), 6.21),
    "384": PlateFormat(16, 24, 4.5, (12.05, 9.05), 3.30),
    "1536": PlateFormat(32, 48, 2.25, (11.01, 7.87), 1.53),
}


def well_name_to_rc(name: str) -> tuple[int, int]:
    """Well label -> 0-based ``(row, col)``.

    Row letters are spreadsheet-style base-26 (``A``=0 … ``Z``=25, ``AA``=26,
    ``AB``=27 …); the trailing number is a 1-based column so ``1`` -> 0.

        >>> well_name_to_rc("A1"), well_name_to_rc("B12"), well_name_to_rc("AA3")
        ((0, 0), (1, 11), (26, 2))

    Raises ``ValueError`` on a name missing the letter or number part.
    """
    s = name.strip().upper()
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    letters, digits = s[:i], s[i:]
    if not letters or not digits or not digits.isdigit():
        raise ValueError(f"malformed well name: {name!r}")
    row = 0
    for ch in letters:
        row = row * 26 + (ord(ch) - ord("A") + 1)
    return (row - 1, int(digits) - 1)


# spacing match tolerance (relative). The plate pitches (2.25 … 39.2 mm) are all
# >25% apart, so any exact grid lands on one format; sloppier inputs fall out as
# "unknown" rather than guessing.
_SPACING_REL_TOL = 0.10
_COORD_EPS_MM = 1e-4  # ignore sub-100nm gaps (fp noise / repeated coordinates)


def _observed_spacing(centers: list[tuple[float, float]]) -> float | None:
    """Smallest positive gap between adjacent well centers along either axis.

    For a contiguous block of wells this equals the plate pitch; taking the
    minimum over per-axis sorted gaps tolerates single-row / single-column
    scans and stray fp noise.
    """
    diffs: list[float] = []
    for axis in (0, 1):
        coords = sorted({round(c[axis], 6) for c in centers})
        diffs += [b - a for a, b in zip(coords, coords[1:]) if b - a > _COORD_EPS_MM]
    return min(diffs) if diffs else None


def infer_plate_format(region_centers_mm: list[tuple[float, float]]) -> str:
    """Guess the plate format key from observed well-center spacing.

    Returns the nearest ``PLATE_FORMATS`` key whose pitch matches the observed
    inter-well spacing within tolerance, or ``"unknown"`` when there are too few
    distinct centers, no format is close, or two formats are equally plausible.
    """
    distinct = {(round(x, 6), round(y, 6)) for x, y in region_centers_mm}
    if len(distinct) < 2:
        return "unknown"
    spacing = _observed_spacing([(float(x), float(y)) for x, y in region_centers_mm])
    if spacing is None:
        return "unknown"

    scored = sorted(
        (abs(fmt.well_spacing_mm - spacing) / fmt.well_spacing_mm, name)
        for name, fmt in PLATE_FORMATS.items()
    )
    best_err, best = scored[0]
    if best_err > _SPACING_REL_TOL:
        return "unknown"
    if len(scored) > 1 and scored[1][0] <= _SPACING_REL_TOL:
        return "unknown"  # ambiguous: another format also matches
    return best


def _as_xy(v: float | tuple[float, float]) -> tuple[float, float]:
    """Scalar -> (v, v); pass a 2-sequence through as (x, y)."""
    if isinstance(v, (tuple, list)):
        return float(v[0]), float(v[1])
    return float(v), float(v)


def _tiles_along(length: float, overlap_percent: float, fov: float) -> int:
    """FOV count to cover ``length`` given per-FOV width ``fov`` and % overlap.

    Step between FOVs is ``fov * (1 - overlap/100)``; ``n`` tiles then span
    ``fov + (n-1)*step``, so ``n = 1 + ceil((length - fov) / step)`` (>= 1 while
    the FOV is positive; degenerate >=100% overlap clamps to a single FOV).
    """
    if fov <= 0:
        return 0
    if length <= fov:
        return 1
    step = fov * (1.0 - overlap_percent / 100.0)
    if step <= 0:
        return 1
    return 1 + math.ceil((length - fov) / step)


def fov_grid_shape(
    scan_size_mm: float | tuple[float, float],
    overlap_percent: float,
    fov_size_mm: float | tuple[float, float],
) -> tuple[int, int]:
    """``(nx, ny)`` FOVs needed to tile a scan region with the given overlap.

    ``scan_size_mm`` and ``fov_size_mm`` may each be a scalar (square) or an
    ``(x, y)`` pair. Overlap is a percentage (``50`` == 50%).
    """
    sx, sy = _as_xy(scan_size_mm)
    fx, fy = _as_xy(fov_size_mm)
    return (_tiles_along(sx, overlap_percent, fx),
            _tiles_along(sy, overlap_percent, fy))


def _check_grid(k: int, nx: int, ny: int) -> None:
    if nx < 1 or ny < 1:
        raise ValueError(f"grid must be positive, got nx={nx}, ny={ny}")
    if not 0 <= k < nx * ny:
        raise IndexError(f"k={k} out of range for {nx}x{ny} grid")


def spattern_index_to_ij(k: int, nx: int, ny: int) -> tuple[int, int]:
    """Serpentine (boustrophedon) scan index ``k`` -> ``(row, col)``.

    Rows fill top to bottom; even rows run left->right, odd rows right->left,
    matching squid's S-pattern acquisition order (``nx`` = columns per row).

        >>> [spattern_index_to_ij(k, 3, 2) for k in range(6)]
        [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    """
    _check_grid(k, nx, ny)
    row, pos = divmod(k, nx)
    col = pos if row % 2 == 0 else nx - 1 - pos
    return (row, col)


def unidirectional_index_to_ij(k: int, nx: int, ny: int) -> tuple[int, int]:
    """Raster scan index ``k`` -> ``(row, col)``; every row runs left->right."""
    _check_grid(k, nx, ny)
    row, col = divmod(k, nx)
    return (row, col)
