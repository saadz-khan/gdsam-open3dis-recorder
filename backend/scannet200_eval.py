#!/usr/bin/env python
"""SECOND BENCHMARK: ScanNet200 class-agnostic mask AP.

Why this benchmark and not another. ScanNet200 re-annotates the same meshes with 198 instance
categories instead of ~18, so it contains the LONG TAIL -- "kitchen counter", "oven", "furniture",
"trash can" -- that a closed-vocabulary method never sees and an open-vocabulary method should own. It
is also the benchmark MaskClustering, SAI3D and Open3DIS actually headline, so winning on ScanNetV2
alone leaves the obvious objection that we chose the easier annotation. Because the mesh is identical,
our existing per-vertex predictions score against it directly: same predictions, same evaluator, a
strictly harder ground truth. No re-clustering, no re-recording, nothing refitted.

INDEX CORRESPONDENCE IS VERIFIED, NOT ASSUMED. The `.pth` files store centred coordinates, so before
any label is trusted we check that the centred `.pth` coordinates reproduce the centred mesh
vertices to within a tight tolerance. A silent index mismatch would produce plausible-looking but
meaningless numbers, which is exactly the failure this guard exists to prevent.

  PYTHONPATH=<repo> HELD_DIR=<recdir> <env>/python scannet200_eval.py --scenes-file scale312_scenes.txt
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
def _env(k, d):
    """Path overridable by environment so this package runs on any machine."""
    return _os.environ.get(k, d)
REPO = _env("C1_REPO", _os.path.dirname(_HERE))
BASE = _env("C1_BASE", _os.path.dirname(_HERE))
sys.path.insert(0, REPO); sys.path.insert(0, BASE); sys.path.insert(0, f"{BASE}/Open3DIS")
if not hasattr(np, "in1d"):
    np.in1d = np.isin
import official_eval as OE
S200 = _env("C1_S200", f"{REPO}/scannet200_val")


def gt200(scene, nV, verify_xyz=None, tol=1e-3):
    """Class-agnostic GT from ScanNet200. Returns (sem, ins, n_inst) in the evaluator's encoding.

    The evaluator computes `gts // encode_value` for the class and `gts % encode_value` for the
    instance, and applies a (-2 + 1) shift to the class id, so a single foreground class must be
    written as semantic id 2 -- the same convention `official_eval.gt_labels` uses for ScanNetV2.
    """
    import torch
    f = f"{S200}/{scene}.pth"
    if not os.path.exists(f):
        return None
    d = torch.load(f, weights_only=False)
    xyz = np.asarray(d[0], np.float64)
    ins200 = np.asarray(d[3]).astype(int)
    if len(ins200) != nV:
        return None                                   # different mesh -> refuse to score
    if verify_xyz is not None:
        a = xyz - xyz.mean(0)
        b = verify_xyz - verify_xyz.mean(0)
        if float(np.abs(a - b).max()) > tol:
            return None                               # index mismatch -> refuse to score
    return collapse_semantic_valid(d[2], d[3])


def collapse_semantic_valid(semantic, instance, min_region_size=100):
    """Retain valid instance geometry while preserving semantic void labels."""
    semantic, instance = np.asarray(semantic), np.asarray(instance)
    if semantic.ndim != 1 or instance.shape != semantic.shape:
        raise ValueError("Semantic and instance channels must be aligned vectors")
    if min_region_size <= 0:
        raise ValueError("Minimum region size must be positive")
    for values in (semantic, instance):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Labels must be finite integers")
    sem = np.zeros(len(instance), np.int64)
    ins = np.full(len(instance), -1, np.int64)
    k = 0
    for ident in np.unique(instance):
        if ident < 0:
            continue
        mask = instance == ident
        valid = semantic[mask] >= 0
        if valid.any() and not valid.all():
            raise ValueError("Instance mixes valid and ignored semantics")
        if not valid.any() or mask.sum() < min_region_size:
            continue
        sem[mask], ins[mask] = 2, k
        k += 1
    return sem, ins, k



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-file", default=f"{BASE}/scale312_scenes.txt")
    ap.add_argument("--methods", default="mc,sai3d,ours")
    ap.add_argument("--score-protocol", choices=("constant", "method-native"), default="constant",
                    help="constant matches published class-agnostic ScanNet200 tables")
    ap.add_argument("--strata", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    scenes = [s.strip() for s in open(args.scenes_file) if s.strip()]
    if args.limit:
        scenes = scenes[: args.limit]
    methods = args.methods.split(",")
    from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval
    ev = ScanNetEval(class_labels=["object"], use_label=False, dataset_name="scannet200")

    store, gts, kept, skipped = {m: {} for m in methods}, {}, [], 0
    for sc in scenes:
        V = OE._read_ply_xyz(f"{OE.SCANS}/{sc}/{sc}_vh_clean_2.ply")
        g = gt200(sc, len(V), verify_xyz=V)
        if g is None:
            skipped += 1; continue
        gts[sc] = g
        for m in methods:
            store[m][sc] = OE._apply_score_protocol(OE.METHODS[m](sc, len(V)), args.score_protocol)
        if all(len(store[m][sc]) > 0 for m in methods):
            kept.append(sc)
    n_gt = sum(gts[s][2] for s in kept)
    print(f"\n  === ScanNet200 class-agnostic mask AP — {len(kept)} scenes, {n_gt} GT instances ===")
    print(f"  score protocol: {args.score_protocol}")
    if skipped:
        print(f"  ({skipped} scenes skipped: mesh or index mismatch — refused rather than guessed)")
    print(f"  {'method':22s} {'#pred':>6s} {'AP':>7s} {'AP50':>7s} {'AP25':>7s} {'AR':>7s}")
    for m in methods:
        preds = [store[m][s] for s in kept]
        avg = ev.evaluate(preds, [gts[s][0] for s in kept], [gts[s][1] for s in kept], exp_path="/tmp")
        print(f"  {OE.NAMES[m]:22s} {sum(len(p) for p in preds):6d} {avg['all_ap']*100:7.1f} "
              f"{avg['all_ap_50%']*100:7.1f} {avg['all_ap_25%']*100:7.1f} {avg['all_rc']*100:7.1f}",
              flush=True)

    for spec in [x for x in args.strata.split(",") if x]:
        name, f = spec.split("=", 1)
        sub = [s for s in (x.strip() for x in open(f)) if s in set(kept)]
        if not sub:
            continue
        print(f"\n  --- {name}: {len(sub)} scenes, {sum(gts[s][2] for s in sub)} GT ---")
        for m in methods:
            preds = [store[m][s] for s in sub]
            avg = ev.evaluate(preds, [gts[s][0] for s in sub], [gts[s][1] for s in sub],
                              exp_path="/tmp")
            print(f"  {OE.NAMES[m]:22s} {sum(len(p) for p in preds):6d} {avg['all_ap']*100:7.1f} "
                  f"{avg['all_ap_50%']*100:7.1f} {avg['all_ap_25%']*100:7.1f} "
                  f"{avg['all_rc']*100:7.1f}", flush=True)


if __name__ == "__main__":
    main()
