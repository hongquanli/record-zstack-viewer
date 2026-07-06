from pathlib import Path
import json, numpy as np, tensorstore as ts


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
                x_mm, y_mm = 10.0 + wi*9 + (k % 3)*0.3, 10.0 + (k // 3)*0.3
                coords.append(f"{wl},{k},0,{x_mm:.3f},{y_mm:.3f},3000.0"
                              + (f",{t}" if time_lapse else ""))
                # ---- recording: one store per plane ----
                ev(type="fov_started", t=t, region=wl, fov=k, phase="recording",
                   x_mm=x_mm, y_mm=y_mm, z_ref_mm=3.0)  # once per (t,region,fov,phase)
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
                    _patch_zarr_json(sd, {"ome": {"multiscales": [{"datasets": [{"path": "."}],
                                     "axes": [{"name": a} for a in "tczyx"]}],
                                     "omero": _omero([("REC", "#FFFFFF")])},
                                     "_squid": squid})
                    if not partial:
                        ev(type="store_finalized", t=t, region=wl, fov=k, phase="recording",
                           path=str(sd), acquisition_complete=True, aborted=False,
                           captured=n_written, expected=recording_frames)
                # ---- z-stack: one store per (well,fov), T axis accumulates ----
                ev(type="fov_started", t=t, region=wl, fov=k, phase="zstack",
                   x_mm=x_mm, y_mm=y_mm, z_ref_mm=3.0)
                grp = root / "zarr" / wl / f"fov_{k}.ome.zarr"
                arrp = grp / "0"
                nc = len(zstack_channels)
                a = (_open_create(arrp, (nt, nc, zstack_nz, h, w), (1, 1, 1, h, w))
                     if not arrp.exists()
                     else ts.open({"driver": "zarr3",
                                   "kvstore": {"driver": "file", "path": str(arrp)}}).result())
                for c in range(nc):
                    for z in range(zstack_nz if not partial else 1):
                        a[t, c, z].write(_synthetic(h, w, seed=9+t+c*3+z)).result()
                grp.mkdir(parents=True, exist_ok=True)
                (grp / "zarr.json").write_text(json.dumps({
                    "zarr_format": 3, "node_type": "group",
                    "attributes": {"ome": {"multiscales": [{"datasets": [{"path": "0"}],
                                   "axes": [{"name": ax} for ax in "tczyx"]}],
                                   "omero": _omero(zstack_channels)},
                                   "_squid": {"structure": "5D-TCZYX", "pixel_size_um": 0.5,
                                   "z_step_um": 1.0, "time_increment_s": 60.0,
                                   "acquisition_complete": (not partial)}}}, indent=2))
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
