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
  - `ome.multiscales[0].datasets[0].coordinateTransformations`: a `scale`
    `{type:scale, scale:[t_s, 1, z_step_um, px_um, px_um]}` (dataset path `.` for
    recording / `0` for z-stack). A per-FOV `translation` is appended — see Placement.
- `acquisition.yaml` (at run START): `wellplate_scan.regions[]{name,center_mm,shape}`,
  `scan_size_mm`, `overlap_percent`, `xy_mode`.
- `acquisition_channels.yaml` (at run START): objective, channels (names+colors).
- `.done` (empty, at run END; also on abort).

## Additive v1.0 (optional; fallbacks noted)
### Per-FOV placement — `coordinateTransformations.translation` (authoritative) + `coordinates.csv` (derived)
**Primary — OME metadata.** Append a `translation` after the `scale` in each store's
`multiscales[0].datasets[0].coordinateTransformations`, placing the FOV in global stage space:
```
"coordinateTransformations": [
  {"type":"scale",       "scale":       [t_s, 1.0, z_step_um, px_um, px_um]},
  {"type":"translation", "translation": [0.0, 0.0, z_stage_um, y_stage_um, x_stage_um]}
]
```
- Values in µm (axis unit); convert stage mm → µm (×1000). Z translation = per-FOV **stage reference Z**.
- A FOV's recording, z-stack, and every `fov_{k}_z{j}` plane store share the **same XY**; each plane's
  Z translation is its absolute stage Z (`z_ref + bottom + j·dz`).
- Standard OME-NGFF 0.4/0.5 (`scale`, then optional `translation` listed after it); read by generic
  NGFF viewers. *Fallback when absent (old data): reconstruct the grid from `acquisition.yaml` geometry.*

**Derived export — `coordinates.csv`** (experiment root): columns `region, fov, z_level, x (mm), y (mm), z (um)`
(+ `time` for time-lapse), one row per FOV, mirroring classic multipoint. Written from the same stage
values; the OME `translation` is authoritative if the two ever disagree.

### events.jsonl  (fallback: file-watch + poll)
Experiment root, append-only, one JSON object/line, each has `type`,`ts`,`format_version`.
- `run_started`: {plate_format, wells:[...], expected_fovs_per_well:{well:int},
  Nt, recording_nz (planes per recording), zstack_nz (z-stack planes),
  channels:[{label,color}], fps, duration_s}
- `fov_started`: {t, region, fov, phase:"recording"|"zstack", x_mm, y_mm, z_ref_mm}
- `heartbeat`: {t, region, fov, phase, written:int}   (throttled)
- `store_finalized`: {t, region, fov, phase, path, acquisition_complete, aborted, captured, expected}
- `timepoint_done`: {t}
- `aborted`: {reason}   |   `run_done`: {}

### thumbnails/  (fallback: compute from a decoded plane)
`thumbnails/{well}/fov_{k}.png` (~128–256 px). Optional running `downsampled/plate.png`.
