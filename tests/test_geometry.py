"""Task 2.1 — pure geometry: well-name math, plate formats, FOV grid + scan patterns.

No Qt / IO / tensorstore here — every symbol under test is a pure function or a
constant table, so these tests are plain arithmetic assertions.
"""
import pytest


# --- well_name_to_rc -------------------------------------------------------

def test_well_name_to_rc_basic():
    from record_zstack_viewer.model.geometry import well_name_to_rc
    assert well_name_to_rc("A1") == (0, 0)
    assert well_name_to_rc("B12") == (1, 11)
    assert well_name_to_rc("AA3") == (26, 2)  # multi-letter row, spreadsheet-style


def test_well_name_to_rc_boundaries():
    from record_zstack_viewer.model.geometry import well_name_to_rc
    assert well_name_to_rc("H12") == (7, 11)   # 96-well last well
    assert well_name_to_rc("Z1") == (25, 0)    # last single-letter row
    assert well_name_to_rc("AB1") == (27, 0)   # AA=26, AB=27
    assert well_name_to_rc("P24") == (15, 23)  # 384-well last well
    assert well_name_to_rc("AF48") == (31, 47)  # 1536-well last well (AF=31)


def test_well_name_to_rc_lowercase_and_whitespace():
    from record_zstack_viewer.model.geometry import well_name_to_rc
    assert well_name_to_rc("b12") == (1, 11)
    assert well_name_to_rc(" C3 ") == (2, 2)


def test_well_name_to_rc_malformed_raises():
    from record_zstack_viewer.model.geometry import well_name_to_rc
    with pytest.raises(ValueError):
        well_name_to_rc("12")   # no row letters
    with pytest.raises(ValueError):
        well_name_to_rc("A")    # no column number
    with pytest.raises(ValueError):
        well_name_to_rc("")


# --- PLATE_FORMATS ---------------------------------------------------------

def test_plate_formats_values():
    from record_zstack_viewer.model.geometry import PLATE_FORMATS
    assert set(PLATE_FORMATS) == {"6", "12", "24", "96", "384", "1536"}

    f96 = PLATE_FORMATS["96"]
    assert (f96.rows, f96.cols) == (8, 12)
    assert f96.well_spacing_mm == 9.0
    assert f96.a1_mm == (11.31, 10.75)
    assert f96.well_size_mm == 6.21

    f6 = PLATE_FORMATS["6"]
    assert (f6.rows, f6.cols) == (2, 3)
    assert f6.well_spacing_mm == 39.2
    assert f6.a1_mm == (24.55, 23.01)
    assert f6.well_size_mm == 34.94

    f1536 = PLATE_FORMATS["1536"]
    assert (f1536.rows, f1536.cols) == (32, 48)
    assert f1536.well_spacing_mm == 2.25


def test_plate_formats_rows_times_cols_is_well_count():
    from record_zstack_viewer.model.geometry import PLATE_FORMATS
    for name, fmt in PLATE_FORMATS.items():
        assert fmt.rows * fmt.cols == int(name)


def test_plate_format_is_immutable():
    from record_zstack_viewer.model.geometry import PLATE_FORMATS
    with pytest.raises(Exception):  # frozen dataclass -> FrozenInstanceError
        PLATE_FORMATS["96"].rows = 1


# --- infer_plate_format ----------------------------------------------------

def _grid_centers(fmt, rows, cols):
    """Well centers (mm) for the top-left rows x cols block of a plate format."""
    ax, ay = fmt.a1_mm
    s = fmt.well_spacing_mm
    return [(ax + c * s, ay + r * s) for r in range(rows) for c in range(cols)]


def test_infer_plate_format_96_vs_384():
    from record_zstack_viewer.model.geometry import infer_plate_format, PLATE_FORMATS
    assert infer_plate_format(_grid_centers(PLATE_FORMATS["96"], 3, 4)) == "96"
    assert infer_plate_format(_grid_centers(PLATE_FORMATS["384"], 3, 4)) == "384"


def test_infer_plate_format_every_format_roundtrips():
    from record_zstack_viewer.model.geometry import infer_plate_format, PLATE_FORMATS
    for name, fmt in PLATE_FORMATS.items():
        centers = _grid_centers(fmt, min(2, fmt.rows), min(3, fmt.cols))
        assert infer_plate_format(centers) == name


def test_infer_plate_format_single_row_or_column():
    from record_zstack_viewer.model.geometry import infer_plate_format, PLATE_FORMATS
    f = PLATE_FORMATS["96"]
    ax, ay = f.a1_mm
    s = f.well_spacing_mm
    assert infer_plate_format([(ax + c * s, ay) for c in range(4)]) == "96"  # one row
    assert infer_plate_format([(ax, ay + r * s) for r in range(4)]) == "96"  # one column


def test_infer_plate_format_unknown():
    from record_zstack_viewer.model.geometry import infer_plate_format
    assert infer_plate_format([]) == "unknown"
    assert infer_plate_format([(0.0, 0.0)]) == "unknown"                 # single center
    assert infer_plate_format([(0.0, 0.0), (0.0, 0.0)]) == "unknown"     # duplicate center
    assert infer_plate_format([(0.0, 0.0), (100.0, 0.0)]) == "unknown"   # no format ~100mm
    # 22.5 mm sits between 24-well (19.3) and 12-well (26.0) -> ambiguous
    assert infer_plate_format([(0.0, 0.0), (22.5, 0.0)]) == "unknown"


# --- fov_grid_shape --------------------------------------------------------

def test_fov_grid_shape_no_overlap():
    from record_zstack_viewer.model.geometry import fov_grid_shape
    assert fov_grid_shape(1.0, 0.0, 1.0) == (1, 1)
    assert fov_grid_shape(2.0, 0.0, 1.0) == (2, 2)
    assert fov_grid_shape(0.5, 0.0, 1.0) == (1, 1)  # scan smaller than one FOV
    assert fov_grid_shape(3.0, 0.0, 1.0) == (3, 3)


def test_fov_grid_shape_overlap():
    from record_zstack_viewer.model.geometry import fov_grid_shape
    # 50% overlap -> step 0.5; cover 2.0 with 1.0 FOVs -> 1 + ceil(1.0/0.5) = 3
    assert fov_grid_shape(2.0, 50.0, 1.0) == (3, 3)
    # 10% overlap -> step 0.9; cover 1.9 -> 1 + ceil(0.9/0.9) = 2
    assert fov_grid_shape(1.9, 10.0, 1.0) == (2, 2)


def test_fov_grid_shape_rectangular():
    from record_zstack_viewer.model.geometry import fov_grid_shape
    assert fov_grid_shape((2.0, 1.0), 0.0, 1.0) == (2, 1)
    assert fov_grid_shape((2.0, 3.0), 0.0, (1.0, 1.0)) == (2, 3)


# --- scan patterns ---------------------------------------------------------

def test_spattern_serpentine():
    from record_zstack_viewer.model.geometry import spattern_index_to_ij
    # 3 wide x 2 tall: row0 L->R, row1 R->L
    assert [spattern_index_to_ij(k, 3, 2) for k in range(6)] == \
        [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]


def test_spattern_three_rows():
    from record_zstack_viewer.model.geometry import spattern_index_to_ij
    # 2 wide x 3 tall: row0 L->R, row1 R->L, row2 L->R
    assert [spattern_index_to_ij(k, 2, 3) for k in range(6)] == \
        [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1)]


def test_unidirectional_raster():
    from record_zstack_viewer.model.geometry import unidirectional_index_to_ij
    assert [unidirectional_index_to_ij(k, 3, 2) for k in range(6)] == \
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_scan_patterns_cover_grid_exactly_once():
    from record_zstack_viewer.model.geometry import (
        spattern_index_to_ij, unidirectional_index_to_ij)
    nx, ny = 5, 4
    for fn in (spattern_index_to_ij, unidirectional_index_to_ij):
        seen = {fn(k, nx, ny) for k in range(nx * ny)}
        assert seen == {(r, c) for r in range(ny) for c in range(nx)}


def test_scan_pattern_bounds_and_validation():
    from record_zstack_viewer.model.geometry import (
        spattern_index_to_ij, unidirectional_index_to_ij)
    with pytest.raises(IndexError):
        spattern_index_to_ij(6, 3, 2)
    with pytest.raises(IndexError):
        unidirectional_index_to_ij(-1, 3, 2)
    with pytest.raises(ValueError):
        spattern_index_to_ij(0, 0, 2)
    with pytest.raises(ValueError):
        unidirectional_index_to_ij(0, 3, 0)
