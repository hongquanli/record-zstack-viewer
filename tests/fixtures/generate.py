from pathlib import Path
import json, os, re, numpy as np, tensorstore as ts


def _open_create(path: Path, shape, chunk, dtype="uint16"):
    return ts.open({
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {
            "shape": list(shape),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunk)}},
            "data_type": dtype,
            "codecs": [{"name": "blosc",
                        "configuration": {"cname": "lz4", "clevel": 1, "shuffle": "shuffle"}}],
            "fill_value": 0,
        },
        "create": True, "delete_existing": True,
    }).result()   # VERIFY codec spelling against installed tensorstore if this errors


def _patch_zarr_json(store_dir: Path, attributes: dict):
    zj = store_dir / "zarr.json"
    meta = json.loads(zj.read_text())
    meta.setdefault("attributes", {}).update(attributes)
    zj.write_text(json.dumps(meta, indent=2))


_AXES = [{"name": "t", "type": "time", "unit": "second"}, {"name": "c", "type": "channel"},
         {"name": "z", "type": "space", "unit": "micrometer"},
         {"name": "y", "type": "space", "unit": "micrometer"},
         {"name": "x", "type": "space", "unit": "micrometer"}]


def _multiscales(path, squid, z_um, y_um, x_um):
    # per-FOV placement (CONTRACT.md): scale MUST precede translation (OME-NGFF rule);
    # scale mirrors the store's _squid values, translation = absolute stage position in um
    return [{"axes": _AXES, "datasets": [{"path": path, "coordinateTransformations": [
        {"type": "scale", "scale": [squid["time_increment_s"], 1.0, squid["z_step_um"],
                                    squid["pixel_size_um"], squid["pixel_size_um"]]},
        {"type": "translation", "translation": [0.0, 0.0, z_um, y_um, x_um]}]}]}]


def _omero(channels):  # channels: list[(label,color_hex)]
    return {"channels": [
        {"label": lbl, "color": col, "active": True,
         "window": {"start": 0, "end": 65535, "min": 0, "max": 65535}}
        for lbl, col in channels]}


def _synthetic(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = (2000*np.exp(-(((yy-h*0.4)**2+(xx-w*0.4)**2)/(2*(h*0.15)**2)))).astype("uint16")
    return (blob + rng.integers(0, 300, size=(h, w))).astype("uint16")


def generate_experiment(root, *, wells, fovs_per_well, nt, recording_frames,
                        recording_planes, zstack_channels, zstack_nz,
                        hw=(64, 64), rich=True, complete=True):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    h, w = hw
    events = []
    def ev(**kw): events.append({**kw, "ts": 0.0, "format_version": "1.0"})
    ev(type="run_started", plate_format="96", wells=wells,
       expected_fovs_per_well={wl: fovs_per_well for wl in wells}, Nt=nt,
       recording_nz=recording_planes, zstack_nz=zstack_nz,
       channels=[{"label": l, "color": c} for l, c in zstack_channels],
       fps=15.0, duration_s=recording_frames/15.0)

    # config sidecars at START
    (root / "acquisition_channels.yaml").write_text(
        "objective: 20x\nchannels:\n" + "".join(f"  - name: {l}\n" for l, _ in zstack_channels))
    regions = "\n".join(f"    - name: {wl}\n      center_mm: [{10.0+i*9}, {10.0}]\n      shape: Square"
                        for i, wl in enumerate(wells))
    (root / "acquisition.yaml").write_text(
        "wellplate_scan:\n  xy_mode: Select Wells\n  scan_size_mm: 1.0\n  overlap_percent: 10\n"
        f"  regions:\n{regions}\n")

    time_lapse = nt > 1  # CONTRACT.md: coordinates.csv "+ time for time-lapse"
    coords = ["region,fov,z_level,x (mm),y (mm),z (um)" + (",time" if time_lapse else "")]
    n_wells = len(wells)
    for t in range(nt):
        for wi, wl in enumerate(wells):
            for k in range(fovs_per_well):
                last = (t == nt-1 and wi == n_wells-1 and k == fovs_per_well-1)
                partial = last and not complete
                x_mm, y_mm, z_ref_mm = 10.0 + wi*9 + (k % 3)*0.3, 10.0 + (k // 3)*0.3, 3.0
                z_ref_um = z_ref_mm * 1000  # OME translations are in um (axis unit)
                coords.append(f"{wl},{k},0,{x_mm:.3f},{y_mm:.3f},{z_ref_um:.1f}"
                              + (f",{t}" if time_lapse else ""))
                # ---- recording: one store per plane ----
                ev(type="fov_started", t=t, region=wl, fov=k, phase="recording",
                   x_mm=x_mm, y_mm=y_mm, z_ref_mm=z_ref_mm)  # once per (t,region,fov,phase)
                for j in range(recording_planes):
                    name = f"fov_{k}.ome.zarr" if recording_planes == 1 else f"fov_{k}_z{j}.ome.zarr"
                    sd = root / "recording" / f"t{t}" / wl / name
                    arr = _open_create(sd, (recording_frames, 1, 1, h, w), (1, 1, 1, h, w))
                    n_written = recording_frames if not partial else max(1, recording_frames // 2)
                    for f in range(n_written):
                        arr[f, 0, 0].write(_synthetic(h, w, seed=t*1000+wi*100+k*10+f)).result()
                    squid = {"structure": "5D-TCZYX", "pixel_size_um": 0.5, "z_step_um": 2.0,
                             "time_increment_s": 1/15.0,
                             "acquisition_complete": (not partial), "captured_frames": n_written,
                             "expected_frames": recording_frames}
                    if recording_planes > 1:
                        squid |= {"plane_index": j, "plane_z_offset_um": -2.0 + j*2.0,
                                  "n_planes": recording_planes}
                    # plane stores share the FOV's XY; Z = z_ref + per-plane offset
                    z_um = z_ref_um + squid.get("plane_z_offset_um", 0.0)
                    _patch_zarr_json(sd, {"ome": {
                                     "multiscales": _multiscales(".", squid, z_um,
                                                                 y_mm*1000, x_mm*1000),
                                     "omero": _omero([("REC", "#FFFFFF")])},
                                     "_squid": squid})
                    if not partial:
                        ev(type="store_finalized", t=t, region=wl, fov=k, phase="recording",
                           path=str(sd), acquisition_complete=True, aborted=False,
                           captured=n_written, expected=recording_frames)
                # ---- z-stack: one store per (well,fov), T axis accumulates ----
                ev(type="fov_started", t=t, region=wl, fov=k, phase="zstack",
                   x_mm=x_mm, y_mm=y_mm, z_ref_mm=z_ref_mm)
                grp = root / "zarr" / wl / f"fov_{k}.ome.zarr"
                arrp = grp / "0"
                nc = len(zstack_channels)
                a = (_open_create(arrp, (nt, nc, zstack_nz, h, w), (1, 1, 1, h, w))
                     if not arrp.exists()
                     else ts.open({"driver": "zarr3",
                                   "kvstore": {"driver": "file", "path": str(arrp)}}).result())
                for c in range(nc):
                    for z in range(zstack_nz if not partial else 1):
                        # mixed-radix seed: distinct multipliers per coordinate so
                        # different (well, fov, t, c, z) yield different pixels —
                        # no cross-well/fov duplicates or diagonal (t,c,z) collisions
                        seed = 9 + wi*1_000_000 + k*100_000 + t*10_000 + c*100 + z
                        a[t, c, z].write(_synthetic(h, w, seed=seed)).result()
                grp.mkdir(parents=True, exist_ok=True)
                zsquid = {"structure": "5D-TCZYX", "pixel_size_um": 0.5,
                          "z_step_um": 1.0, "time_increment_s": 60.0,
                          "acquisition_complete": (not partial)}
                (grp / "zarr.json").write_text(json.dumps({
                    "zarr_format": 3, "node_type": "group",
                    "attributes": {"ome": {  # same XY as this FOV's recording store(s)
                                   "multiscales": _multiscales("0", zsquid, z_ref_um,
                                                               y_mm*1000, x_mm*1000),
                                   "omero": _omero(zstack_channels)},
                                   "_squid": zsquid}}, indent=2))
                if not partial:  # partial store never finalizes (same as recording phase)
                    ev(type="store_finalized", t=t, region=wl, fov=k, phase="zstack",
                       path=str(grp), acquisition_complete=True, aborted=False,
                       captured=nc*zstack_nz, expected=nc*zstack_nz)
        if complete or t < nt - 1:  # partial final timepoint: producer hasn't emitted it yet
            ev(type="timepoint_done", t=t)

    if complete:
        ev(type="run_done")
        (root / ".done").write_text("")
    if rich:
        (root / "coordinates.csv").write_text("\n".join(coords) + "\n")
        (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        thumbs = root / "thumbnails"
        for wl in wells:
            (thumbs / wl).mkdir(parents=True, exist_ok=True)
            for k in range(fovs_per_well):
                _write_png(thumbs / wl / f"fov_{k}.png", _synthetic(32, 32, k))
    return root


def _write_png(path: Path, arr8):
    # minimal 16-bit->8-bit PNG via imageio if available, else numpy .npy fallback
    try:
        import imageio.v3 as iio
        iio.imwrite(path, (arr8 / 256).astype("uint8"))
    except Exception:
        np.save(path.with_suffix(".npy"), arr8)


# --- incremental growth (Task 6.2) -------------------------------------------
# Drive a partial experiment forward on disk, one pending plane per store per
# call, the way a live acquisition fills its pre-declared stores over time.

_GROW_FOV_RE = re.compile(r"^fov_(\d+)(?:_z\d+)?\.ome\.zarr$")


def _digit_dirs(d: Path) -> set:
    """The digit-named children of a zarr3 `c/` chunk dir (its chunk indices)."""
    try:
        return {int(p.name) for p in d.iterdir() if p.name.isdigit()}
    except (FileNotFoundError, NotADirectoryError):
        return set()


def _store_complete(store: Path) -> bool:
    """True once `store`'s `_squid.acquisition_complete` is set (finalized)."""
    try:
        meta = json.loads((store / "zarr.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(((meta.get("attributes") or {}).get("_squid") or {}).get("acquisition_complete"))


def _finalize_atomic(store: Path, **squid_updates) -> None:
    """Merge `squid_updates` into `store/zarr.json`'s `_squid`, written atomically
    (temp file + os.replace) so a reader never observes a torn zarr.json."""
    zj = store / "zarr.json"
    meta = json.loads(zj.read_text())
    meta.setdefault("attributes", {}).setdefault("_squid", {}).update(squid_updates)
    tmp = zj.parent / (zj.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, zj)                                  # atomic swap within the dir


def grow_experiment(root, step):
    """Advance a partially-written experiment on disk by one growth `step`.

    Emulates one increment of a live acquisition: for every not-yet-finalized
    store under `root`, write its next single pending plane — one recording
    frame (T axis) and, per channel, one z-plane of the z-stack at the current
    (t0) timepoint — filled with a distinct non-zero value ``1000 * (step + 1)``
    so a viewer can tell a freshly-written plane from an unwritten fill (0). A
    store that thereby reaches its full declared extent is finalized
    (``_squid.acquisition_complete = True``; recording also bumps
    ``captured_frames``). Chunk writes are atomic (tensorstore) and the
    ``zarr.json`` finalize is an atomic replace. Growth is one plane per store
    per call, so a caller can drive ``LiveEngine.tick()`` between calls and watch
    coverage rise and pixels appear. Returns the set of ``(well, fov)`` whose
    stores were touched (grown or finalized) this step."""
    root = Path(root)
    value = 1000 * (int(step) + 1)
    touched: set = set()

    # -- recording: recording/t{t}/{well}/fov_{k}[_z{j}].ome.zarr (array at root) --
    rec = root / "recording"
    if rec.is_dir():
        for store in sorted(rec.glob("t*/*/fov_*.ome.zarr")):
            m = _GROW_FOV_RE.match(store.name)
            if not (m and store.is_dir()) or _store_complete(store):
                continue
            arr = ts.open({"driver": "zarr3",
                           "kvstore": {"driver": "file", "path": str(store)}}).result()
            total, (h, w) = arr.shape[0], arr.shape[-2:]
            written = _digit_dirs(store / "c")
            nxt = next((f for f in range(total) if f not in written), None)
            if nxt is not None:
                arr[nxt, 0, 0].write(np.full((h, w), value, dtype="uint16")).result()
                written.add(nxt)
            _finalize_atomic(store, acquisition_complete=len(written) >= total,
                             captured_frames=len(written))
            touched.add((store.parent.name, int(m.group(1))))

    # -- z-stack: zarr/{well}/fov_{k}.ome.zarr (array at /0, one z per channel) --
    zroot = root / "zarr"
    if zroot.is_dir():
        for grp in sorted(zroot.glob("*/fov_*.ome.zarr")):
            m = _GROW_FOV_RE.match(grp.name)
            if not (m and grp.is_dir()) or _store_complete(grp):
                continue
            arrp = grp / "0"
            a = ts.open({"driver": "zarr3",
                         "kvstore": {"driver": "file", "path": str(arrp)}}).result()
            _nt, nc, nz, h, w = a.shape
            t = 0                                        # partial fixtures are single-t
            complete = True
            for c in range(nc):
                written = _digit_dirs(arrp / "c" / str(t) / str(c))
                nxt = next((z for z in range(nz) if z not in written), None)
                if nxt is not None:
                    a[t, c, nxt].write(np.full((h, w), value, dtype="uint16")).result()
                    written.add(nxt)
                complete = complete and len(written) >= nz
            if complete:
                _finalize_atomic(grp, acquisition_complete=True)
            touched.add((grp.parent.name, int(m.group(1))))

    return touched
