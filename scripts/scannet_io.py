#!/usr/bin/env python
"""ScanNet v2 I/O: axis-aligned GT instance boxes + a minimal .sens RGB-D reader.

GT instances come from the official mesh + over-segmentation + aggregation, put into the
axis-aligned world frame (the standard ScanNet 3D-detection convention, so GT boxes are AABBs
and the Manhattan room frame is the identity). The .sens reader yields real captured RGB,
sensor depth, camera-to-world pose and depth intrinsics per frame -- the real RGB-D the
ClutterTwins pipeline is meant to run on.
"""
from __future__ import annotations
import json
import os
import struct
import zlib

import numpy as np

# labels that are scene structure, not placeable objects
STRUCTURAL = {"wall", "floor", "ceiling", "window", "door", "doorframe", "doors", "curtain",
              "stairs", "railing", "column", "beam", "kitchen cabinets", "ceiling light",
              "object", "remove", "unknown"}


def load_axis_align(txt_path):
    M = np.eye(4)
    for line in open(txt_path):
        if line.startswith("axisAlignment"):
            M = np.array([float(x) for x in line.split("=")[1].split()], float).reshape(4, 4)
    return M


def _read_ply_xyz(path):
    """Read vertex (x,y,z) from a binary_little_endian ScanNet _vh_clean_2.ply."""
    with open(path, "rb") as f:
        n_verts = 0; props = []; in_vertex = False
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1]); in_vertex = True
            elif line.startswith("element") and not line.startswith("element vertex"):
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        # property types: x,y,z float; then red,green,blue,alpha uchar (ScanNet)
        names = props
        fmt = ""; size = 0
        tmap = {"float": ("f", 4), "uchar": ("B", 1), "uint8": ("B", 1), "int": ("i", 4)}
        # infer from standard ScanNet layout: x y z (float) red green blue alpha (uchar)
        rec = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")])
        data = np.frombuffer(f.read(n_verts * rec.itemsize), dtype=rec, count=n_verts)
    return np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)


def load_gt_instances(scene_dir, scene, up=np.array([0.0, 0.0, 1.0]), min_verts=200):
    """Axis-aligned GT object instances: list of dict(label, center, size, yaw=0, verts)."""
    M = load_axis_align(os.path.join(scene_dir, f"{scene}.txt"))
    V = _read_ply_xyz(os.path.join(scene_dir, f"{scene}_vh_clean_2.ply"))
    Va = (np.c_[V, np.ones(len(V))] @ M.T)[:, :3]                     # axis-aligned world
    segs = json.load(open(os.path.join(scene_dir, f"{scene}_vh_clean_2.0.010000.segs.json")))
    seg_idx = np.asarray(segs["segIndices"])
    agg = json.load(open(os.path.join(scene_dir, f"{scene}.aggregation.json")))
    out = []
    for g in agg["segGroups"]:
        lab = str(g["label"]).lower()
        if lab in STRUCTURAL:
            continue
        mask = np.isin(seg_idx, np.asarray(g["segments"]))
        if mask.sum() < min_verts:
            continue
        P = Va[mask]
        lo, hi = P.min(0), P.max(0)
        out.append({"label": lab, "objectId": g.get("objectId", g["id"]),
                    "center": (lo + hi) / 2.0, "size": (hi - lo),
                    "yaw": 0.0, "verts": P})
    return out, M


# ----------------------------- .sens reader -----------------------------

def read_sens(path, stride=20, max_frames=120):
    """Yield (rgb HxWx3 uint8, depth_m HxW float, cam2world 4x4, Kdepth dict) for sampled frames."""
    import imageio.v2 as imageio
    with open(path, "rb") as f:
        struct.unpack("I", f.read(4))                                # version
        strlen = struct.unpack("Q", f.read(8))[0]; f.read(strlen)    # sensor name
        f.read(16 * 4 * 4)                                           # 4 x 4x4 intrinsics/extrinsics
        f.read(4)                                                    # color compression type
        f.read(4)                                                    # depth compression type
        cw, ch, dw, dh = struct.unpack("IIII", f.read(16))
        depth_shift = struct.unpack("f", f.read(4))[0]
        n_frames = struct.unpack("Q", f.read(8))[0]
        # depth intrinsics are in the .txt; here we just stream frames
        idxs = set(range(0, n_frames, stride))
        if len(idxs) > max_frames:
            idxs = set(np.linspace(0, n_frames - 1, max_frames).astype(int).tolist())
        for i in range(n_frames):
            c2w = np.frombuffer(f.read(16 * 4), np.float32).reshape(4, 4).astype(np.float64)
            f.read(16)                                              # 2 timestamps (uint64 each)
            csz = struct.unpack("Q", f.read(8))[0]
            dsz = struct.unpack("Q", f.read(8))[0]
            cdata = f.read(csz); ddata = f.read(dsz)
            if i not in idxs or not np.isfinite(c2w).all():
                continue
            rgb = imageio.imread(cdata)[:, :, :3]                    # jpeg color
            depth = np.frombuffer(zlib.decompress(ddata), np.uint16).reshape(dh, dw).astype(np.float64)
            depth /= depth_shift                                    # -> metres
            yield rgb, depth, c2w, (cw, ch, dw, dh)
