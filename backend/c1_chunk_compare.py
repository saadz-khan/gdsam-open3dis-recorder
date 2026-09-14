#!/usr/bin/env python
"""Three Grounded-SAM banks on the same scenes: ours, Open3DIS chunk 10, Open3DIS chunk 1.

This is the only comparison that isolates the query protocol itself. All three banks are
Grounded-SAM at stride 10 / 200 frames, fed to the identical backend at bank floor 0.55 and scored
by the same untouched evaluator at constant confidence 1.0.

  v5_gdsam_v96vocab   ours: 96 concepts, thresholds not recorded
  gdsam_o3d_c10       Open3DIS config, 198 classes, 10 classes per query
  gdsam_node/out      Open3DIS config, 198 classes, ONE class per query (their literal protocol)

AP over a handful of scenes is close to meaningless -- it is a set-level quantity and a two-scene
sample moves several points on one instance -- so reach@0.25 and mask count are reported alongside.
Reach is a per-instance property of the front end alone and is the stable signal at this sample size.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

B = "/home/saad/Desktop/spacesculptor_c1ap25"
R = "/home/saad/Desktop/spacesculptor_old"
sys.path[:0] = [B, R]

BANKS = [("ours (96 concepts)", f"{R}/runs/scannet_ap/v5_gdsam_v96vocab"),
         ("Open3DIS cfg, chunk 10", f"{R}/runs/scannet_ap/gdsam_o3d_c10"),
         ("Open3DIS cfg, chunk 1", f"{R}/tools/gdsam_node/out")]


def reach(rec_path, scene, oe, annotation, T=0.25):
    rec = pickle.load(open(rec_path, "rb"))
    nV = rec["P"].shape[0]
    Pc = rec["P"].tocsc()
    if annotation == "scannetv2":
        sem, ins, k = oe.gt_labels(scene, nV, "scannetv2")
    else:
        from scannet200_eval import gt200
        got = gt200(scene, nV)
        if got is None:
            return 0, 0, rec["nM"]
        sem, ins, k = got
    hit = tot = 0
    for g in range(k):
        gm = np.flatnonzero(ins == g)
        if len(gm) < 100:
            continue
        tot += 1
        gs = np.zeros(nV, bool)
        gs[gm] = True
        for c in range(Pc.shape[1]):
            vv = Pc.indices[Pc.indptr[c]:Pc.indptr[c + 1]]
            inter = int(gs[vv].sum())
            if inter and inter / (len(gm) + len(vv) - inter) >= T:
                hit += 1
                break
    return hit, tot, rec["nM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="scannetv2")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import official_eval as oe

    common = None
    for _, d in BANKS:
        s = {p.stem for p in Path(d).glob("*.pkl")}
        common = s if common is None else (common & s)
    scenes = sorted(common)
    if not scenes:
        print("  no scene is present in all three banks yet")
        return
    lst = Path("/tmp/claude-1001/chunk_cmp_scenes.txt")
    lst.write_text("\n".join(scenes) + "\n")
    print(f"  {len(scenes)} scene(s) in all three banks: {', '.join(scenes)}\n")

    print(f"  {'bank':26s} {'masks/scene':>12s} {'reach@.25':>10s} {'GT':>5s}")
    for name, d in BANKS:
        h = t = m = 0
        for sc in scenes:
            hh, tt, mm = reach(f"{d}/{sc}.pkl", sc, oe, a.annotation)
            h += hh
            t += tt
            m += mm
        print(f"  {name:26s} {m/len(scenes):12.0f} {100*h/max(t,1):9.1f}% {t:5d}", flush=True)

    for name, d in BANKS:
        print(f"\n  ######## {name} ########", flush=True)
        subprocess.run([sys.executable, f"{B}/c1_joint_bank.py", "--frontend", "gdsam",
                        "--rec-dir", d, "--scans", f"{R}/datasets/scannet_raw/scans",
                        "--scenes-file", str(lst), "--annotation", a.annotation,
                        "--floors", "0.55", "--limit", "0",
                        "--workers", str(a.workers)], cwd=B)


if __name__ == "__main__":
    main()
