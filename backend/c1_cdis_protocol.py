#!/usr/bin/env python
"""Score our frozen predictions under CDIS's protocol, with CDIS's ground truth, exactly.

The comparison was indeterminate because CDIS's repository ships the evaluator but not the GT it
reads.  DATA.md names the format -- "the openmask3d instance-GT format ... semantic_label * 1000 +
instance_id" -- and OpenMask3D's generator is on disk, so the file is reconstructible rather than
guessable.  From `third_party/openmask3d/.../datasets/preprocessing/scannet_preprocessing.py`:

    labels[occupied, 0] = semantic id of the aggregation group
    labels[occupied, 1] = instance["id"]            (-1 where no group covers the point)
    gt_data = points[:, -2] * 1000 + points[:, -1] + 1

Two semantic vocabularies reach that first column.  The ScanNet200 branch writes the RAW category id
from scannetv2-labels.combined.tsv; the base branch writes what `_vh_clean_2.labels.ply` holds, the
NYU40 id.  Both are built here because CDIS's config names `scannetv2` as the dataset and
`scannet200` as the experiment, and its README reports a ScanNet200 row.  It does not matter much:
raw id 1 and NYU40 id 1 are both "wall", and in class-agnostic mode the evaluator sets
VALID_CLASS_IDS = [1], so under either vocabulary the only non-void surface in the whole scene is
wall.

Our own protocol is scored alongside as the reference.  The predictions are identical in every row;
the ONLY thing that changes is which ground truth file the evaluator opens.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

B = "/home/saad/Desktop/spacesculptor_c1ap25"
CDIS = "/tmp/claude-1001/CDIS"
sys.path[:0] = [B, "/home/saad/Desktop/spacesculptor_old"]

_TSV = None


def label_maps():
    """raw_category -> (raw id, nyu40 id), from ScanNet's own label table."""
    global _TSV
    if _TSV is None:
        import official_eval as oe
        raw, nyu = {}, {}
        with open(oe.LABEL_MAP, newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                k = row["raw_category"].strip().lower()
                raw[k] = int(row["id"])
                nyu[k] = int(row["nyu40id"])
        _TSV = (raw, nyu)
    return _TSV


def openmask3d_gt(scene, nV, vocabulary):
    """The exact OpenMask3D instance-GT array: sem * 1000 + instance["id"] + 1, 0 where unannotated."""
    import official_eval as oe
    raw, nyu = label_maps()
    table = raw if vocabulary == "scannet200" else nyu
    segs = json.load(open(f"{oe.SCANS}/{scene}/{scene}_vh_clean_2.0.010000.segs.json"))
    seg_idx = np.asarray(segs["segIndices"])[:nV]
    agg = json.load(open(f"{oe.SCANS}/{scene}/{scene}.aggregation.json"))
    sem = np.zeros(nV, np.int64)
    ins = np.full(nV, -1, np.int64)
    for g in agg["segGroups"]:
        m = np.isin(seg_idx, np.asarray(g["segments"]))
        sem[m] = table.get(str(g["label"]).strip().lower(), 0)
        ins[m] = int(g["id"])
    return sem * 1000 + ins + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/claude-1001/conf312/v96")
    ap.add_argument("--scenes-file", default=f"{B}/all312_scenes.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--work", default="/tmp/claude-1001/cdisgt")
    a = ap.parse_args()

    import official_eval as oe

    scenes = [s.strip() for s in open(a.scenes_file) if s.strip()]
    cache = Path(a.cache)
    scenes = [s for s in scenes if (cache / f"{s}.pkl").exists()][:a.limit or None]

    # predictions, frozen, before any ground truth is opened
    masks = {}
    for sc in scenes:
        nV, out = pickle.load(open(cache / f"{sc}.pkl", "rb"))
        ms = []
        for v, _ in out:
            v = v[(v >= 0) & (v < nV)]
            m = np.zeros(nV, bool)
            m[v] = True
            ms.append(m)
        masks[sc] = ms
    print(f"  {len(scenes)} scenes, {sum(len(masks[s]) for s in scenes):,} frozen predictions")

    work = Path(a.work)
    variants = ("scannet200", "nyu40")
    for v in variants:
        shutil.rmtree(work / v, ignore_errors=True)
        (work / v).mkdir(parents=True, exist_ok=True)

    sems, inss, n_ours = [], [], 0
    counts = {v: 0 for v in variants}
    for sc in scenes:
        nV = len(masks[sc][0])
        sem, ins, k = oe.gt_labels(sc, nV, "scannetv2")
        sems.append(sem)
        inss.append(ins)
        n_ours += k
        for v in variants:
            g = openmask3d_gt(sc, nV, v)
            np.savetxt(work / v / f"{sc}.txt", g, fmt="%d")
            u, c = np.unique(g[g > 0], return_counts=True)
            counts[v] += int(((u >= 1000) & (c >= 100)).sum())
    print(f"  eligible GT instances -- ours (18-class ScanNetV2): {n_ours:,}; "
          + "; ".join(f"OpenMask3D/{v}: {counts[v]:,}" for v in variants))

    rows = {}
    sys.path.insert(0, "/home/saad/Desktop/spacesculptor_baselines/Open3DIS")
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval
    ev = ScanNetEval(class_labels=["object"], use_label=False, dataset_name="scannetv2")
    pl = [[{"scan_id": sc, "label_id": 1, "pred_mask": m, "conf": 1.0} for m in masks[sc]]
          for sc in scenes]
    with contextlib.redirect_stdout(io.StringIO()):
        r = ev.evaluate(pl, sems, inss, exp_path=str(work))
    rows["ours: 18-class GT, our evaluator"] = (r["all_ap"] * 100, r["all_ap_50%"] * 100,
                                                r["all_ap_25%"] * 100)

    sys.path.insert(0, f"{CDIS}/evaluation")
    import eval_semantic_instance as evi
    preds = {}
    for sc in scenes:
        M = np.stack(masks[sc], axis=1).astype(np.uint8)
        preds[sc] = {"pred_masks": M, "pred_scores": np.ones(M.shape[1], np.float32),
                     "pred_classes": np.ones(M.shape[1], np.int64)}
    for v in variants:
        with contextlib.redirect_stdout(io.StringIO()):
            avgs, _, _, _ = evi.evaluate(preds, str(work / v),
                                         output_file=str(work / f"{v}.txt"), class_agnostic=True)
        rows[f"CDIS protocol: OpenMask3D GT ({v})"] = (avgs["all_ap"] * 100,
                                                       avgs["all_ap_50%"] * 100,
                                                       avgs["all_ap_25%"] * 100)

    print(f"\n  {'protocol':40s} {'AP':>8s} {'AP50':>8s} {'AP25':>8s}")
    for k, v in rows.items():
        print(f"  {k:40s} {v[0]:8.3f} {v[1]:8.3f} {v[2]:8.3f}")
    print(f"\n  {'CDIS repository README, 312 S200 scenes':40s} {35.8:8.1f} {54.6:8.1f} {69.9:8.1f}")
    for v in variants:
        o = rows[f"CDIS protocol: OpenMask3D GT ({v})"]
        print(f"  {'  -> our margin under that protocol (' + v + ')':40s} "
              f"{o[0]-35.8:+8.3f} {o[1]-54.6:+8.3f} {o[2]-69.9:+8.3f}")


if __name__ == "__main__":
    main()
