# record-zstack-viewer data contract — v1.0

Producer: Squid Record + Z-Stack (PR #564/#579). Consumer: record-zstack-viewer.
All paths relative to `{experiment}/`. Additive sidecars are OPTIONAL; the viewer
has a fallback for each. `format_version` gates changes.

## Stable (already produced by squid — verified)
- `recording/t{t}/{well}/fov_{k}.ome.zarr` (single plane) or `fov_{k}_z{j}.ome.zarr`
  (multi-plane). Array at STORE ROOT. Shape `(T,1,1,Y,X)`, T=video frames. 1 channel.
- `zarr/{well}/fov_{k}.ome.zarr/0`. Array at `/0` (group metadata one level up).
  Shape `(Nt,C,NZ,Y,X)`, Nt=acquisition timepoints.
- Zarr v3, OME-NGFF 0.5, TCZYX. Chunk `(1,1,1,Y,X)` (FULL_FRAME default) or tiled.
  Compression blosc-lz4 (FAST default; sharded under BALANCED/BEST). dtype uint16/uint8.
  `zarr.json` declares full shape at creation; unwritten chunks read as fill_value 0.
- Per-store `zarr.json attributes`:
  - `ome.omero.channels[]`: {label, color(hex), window{start,end,min,max}, active,
    emission_wavelength?}. NOTE window = full dtype range, NOT a display default.
  - `_squid`: {structure:"5D-TCZYX", pixel_size_um, z_step_um, time_increment_s,
    acquisition_complete(bool), aborted?(bool), captured_frames?, expected_frames?}.
    Multi-plane recording stores add {plane_index, plane_z_offset_um, n_planes};
    absence ⇒ single plane (n_planes=1, plane_index=0).
- `acquisition.yaml` (at run START): `wellplate_scan.regions[]{name,center_mm,shape}`,
  `scan_size_mm`, `overlap_percent`, `xy_mode`.
- `acquisition_channels.yaml` (at run START): objective, channels (names+colors).
- `.done` (empty, at run END; also on abort).

## Additive v1.0 (optional; fallbacks noted)
### coordinates.csv  (fallback: reconstruct from acquisition.yaml + geometry)
Experiment root. Columns: `region, fov, z_level, x (mm), y (mm), z (um)` (+ `time`
for time-lapse). One row per FOV. Mirrors classic multipoint.

### events.jsonl  (fallback: file-watch + poll)
Experiment root, append-only, one JSON object/line, each has `type`,`ts`,`format_version`.
- `run_started`: {plate_format, wells:[...], expected_fovs_per_well:{well:int},
  Nt, Nz, channels:[{label,color}], fps, duration_s}
- `fov_started`: {t, region, fov, phase:"recording"|"zstack", x_mm, y_mm, z_ref_mm}
- `heartbeat`: {t, region, fov, phase, written:int}   (throttled)
- `store_finalized`: {t, region, fov, phase, path, acquisition_complete, aborted, captured, expected}
- `timepoint_done`: {t}
- `aborted`: {reason}   |   `run_done`: {}

### thumbnails/  (fallback: compute from a decoded plane)
`thumbnails/{well}/fov_{k}.png` (~128–256 px). Optional running `downsampled/plate.png`.
