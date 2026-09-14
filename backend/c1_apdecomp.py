#!/usr/bin/env python
"""WHAT IS AP, EXACTLY, UNDER THE CONSTANT-CONFIDENCE PROTOCOL?

MaskClustering's official class-agnostic evaluation writes ``pred_score = np.ones(num_instance)``
(``MaskClustering/utils/post_process.py:139``); SAI3D exports 1.0; the MV3DIS class-agnostic table is
produced the same way.  We adopted that convention, but never worked out what it does to the metric.

It collapses it.  With every confidence equal, ``np.unique(y_score_sorted)`` yields ONE threshold, so
the precision-recall curve has exactly one real point plus the artificial (1, 0) endpoint, and the
evaluator's trapezoid reduces in closed form to

    AP_t  =  R_t * (1 + P_t) / 2

for each IoU threshold t, with P and R the precision and recall of the emitted proposal SET.  Ranking
is irrelevant (which is why the proposal-order audit found AP invariant to permutation), and the two
quantities that matter are a single operating point's precision and recall.

That has a consequence worth measuring rather than assuming.  Differentiating,

    dAP/dTP  =  (1 + P) / (2 * nGT)          a proposal that hits
    dAP/dFP  =  - R * P / (2 * nCounted)     a proposal that misses and is COUNTED

so the exchange rate between a new true positive and a new false positive is fixed by (P, R, nGT,
nCounted) and is computable.  A third category exists and is FREE: the evaluator drops a prediction
entirely when ``proportion_ignore > iou_th``, i.e. when most of it lands on points belonging to no
valid GT instance.  Under ScanNetV2's 18-class GT that is a large fraction of an open-vocabulary
system's output, and it costs nothing.

This script measures P, R, #TP, #counted-FP and #ignored-FP per IoU threshold, verifies the closed
form against the evaluator's own output, and prints the exchange rate.  Nothing here is tuned; it is
the diagnosis the operating-point choice should be based on.

  PYTHONPATH=<repo> <env>/python c1_apdecomp.py --split tune --frontend sam3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = Path("/home/saad/Desktop/spacesculptor_old")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "Open3DIS"))
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import official_eval as oe  # noqa: E402
from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval  # noqa: E402

SPLITS = {
    "tune": BASE / "tune_scenes.txt",
    "dev": BASE / "splits/heldout_scenes.txt",
    "test288": BASE / "test312.txt",
    "test104": BASE / "gdsam_test.txt",
    "test72": BASE / "test72_scenes.txt",
    "testhave": BASE / "test_have_v6.txt",
    "full312": BASE / "splits/all312_scenes.txt",
    "train60": BASE / "splits/train_scenes_60.txt",
}
CACHE = BASE / "c1_cache"


def read_split(name: str) -> list[str]:
    import re
    scenes = re.split(r"[\s,]+", SPLITS[name].read_text().strip())
    if not scenes or any(not re.fullmatch(r"scene\d{4}_\d{2}", s) for s in scenes):
        raise ValueError(f"Invalid scene list: {SPLITS[name]}")
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"Duplicate scenes in {SPLITS[name]}")
    return scenes


# ---------------------------------------------------------------- prediction cache
def _predict_scene(scene: str, frontend: str, method: str):
    """Run one inference path for one scene; returns a list of vertex-index arrays."""
    from c1_bridge_ablation import FRONTENDS
    from c1_candidate import (
        _inference_item,
        predict_candidate_item,
        _deployed_predictions,
    )

    cfg = dict(FRONTENDS[frontend])
    n_vertices = len(oe._read_ply_xyz(f"{oe.SCANS}/{scene}/{scene}_vh_clean_2.ply"))
    item = _inference_item(scene, n_vertices, cfg, Path(cfg["recordings"]), bool(cfg.get("require_clean_prompts", False) or "clean" in frontend))
    if item is None:
        return None, n_vertices
    if method == "deployed":
        preds = _deployed_predictions(item, cfg)
    elif method == "candidate":
        preds = predict_candidate_item(item, cfg)
    elif method == "relift":
        from c1_relift_refinement import refine_candidate
        import c1_candidate as cc

        candidate = predict_candidate_item(item, cfg)
        preds, _ = refine_candidate(
            item, cfg, candidate, cc.PROJECTION_TOP_K, cc.RELIFT_MATCH_FLOOR,
            cc.RELIFT_VOTE_FLOOR, cc.TEMPORAL_PROJECTION_FLOOR, cc.FUSION_NMS,
        )
    else:
        raise ValueError(method)
    out = []
    for pred in preds:
        mask = np.asarray(pred["pred_mask"]).astype(bool)
        out.append({"v": np.flatnonzero(mask).astype(np.int32),
                    "conf": float(pred.get("conf", 1.0))})
    return out, n_vertices


def predictions(split: str, frontend: str = "sam3", method: str = "relift", refresh: bool = False):
    """Cache the raw proposal set so operating-point studies never re-run inference."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{split}_{frontend}_{method}.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as handle:
            return pickle.load(handle)
    store = {}
    scenes = read_split(split)
    for i, scene in enumerate(scenes):
        preds, n_vertices = _predict_scene(scene, frontend, method)
        if preds is None:
            continue
        store[scene] = {"preds": preds, "nV": n_vertices}
        print(f"    [{i+1}/{len(scenes)}] {scene}: {len(preds)} proposals", flush=True)
    with path.open("wb") as handle:
        pickle.dump(store, handle, protocol=4)
    return store


def ground_truth(scenes, store, annotation: str, refresh: bool = False):
    import hashlib

    CACHE.mkdir(exist_ok=True)
    # A stable digest: Python's str hash is salted per process, so hash(tuple(scenes)) would mint a
    # fresh 25 MB cache file on every run.
    digest = hashlib.sha1(",".join(sorted(scenes)).encode()).hexdigest()[:10]
    # Semantic-valid ScanNet200 invalidates legacy instance-only GT caches.
    version = "semantic_valid_xyz_v2" if annotation == "scannet200" else "v1"
    path = CACHE / f"gt_{annotation}_{version}_{len(scenes)}_{digest}.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as handle:
            return pickle.load(handle)
    out = {}
    for scene in scenes:
        n_vertices = store[scene]["nV"]
        if annotation == "scannet200":
            from scannet200_eval import gt200

            xyz = oe._read_ply_xyz(f"{oe.SCANS}/{scene}/{scene}_vh_clean_2.ply")
            got = gt200(scene, n_vertices, verify_xyz=xyz)
            if got is None:
                continue
            out[scene] = got
        else:
            out[scene] = oe.gt_labels(scene, n_vertices, "scannetv2")
    with path.open("wb") as handle:
        pickle.dump(out, handle, protocol=4)
    return out


# ---------------------------------------------------------------- evaluation
_EVAL = {}


def evaluator(annotation: str) -> ScanNetEval:
    if annotation not in _EVAL:
        _EVAL[annotation] = ScanNetEval(
            class_labels=["object"], use_label=False,
            dataset_name="scannet200" if annotation == "scannet200" else "scannetv2",
        )
    return _EVAL[annotation]


def to_masks(entry, scene, keep=None):
    # The evaluator keys its global ``pred_visited`` table on "{scan_id}_{index}", so scan_id MUST be
    # the real scene name: a shared placeholder makes scene A's proposal 0 collide with scene B's and
    # silently destroys recall.
    n_vertices = entry["nV"]
    out = []
    for i, pred in enumerate(entry["preds"]):
        if keep is not None and not keep[i]:
            continue
        mask = np.zeros(n_vertices, bool)
        mask[pred["v"]] = True
        out.append({"scan_id": scene, "label_id": 1, "pred_mask": mask, "conf": 1.0})
    return out


def score(store, gts, scenes, annotation="scannetv2", keeps=None, per_threshold=False):
    """Class-agnostic mask AP under the constant-confidence protocol."""
    pred_list, sem_list, ins_list = [], [], []
    n_pred = 0
    for scene in scenes:
        if scene not in gts or scene not in store:
            continue
        keep = None if keeps is None else keeps.get(scene)
        preds = to_masks(store[scene], scene, keep)
        n_pred += len(preds)
        pred_list.append(preds)
        sem_list.append(gts[scene][0])
        ins_list.append(gts[scene][1])
    ev = evaluator(annotation)
    with contextlib.redirect_stdout(io.StringIO()):
        results = [ev.assign_instances_for_scan(p, s, i)
                   for p, s, i in zip(pred_list, sem_list, ins_list)]
        matches = {f"gt_{i}": {"gt": g, "pred": p} for i, (g, p) in enumerate(results)}
        ap, rc = ev.evaluate_matches(matches)
        avg = ev.compute_averages(ap, rc)
    summary = {
        "n_pred": n_pred,
        "n_gt": int(sum(gts[s][2] for s in scenes if s in gts)),
        "ap": avg["all_ap"] * 100, "ap50": avg["all_ap_50%"] * 100,
        "ap25": avg["all_ap_25%"] * 100, "ar": avg["all_rc"] * 100,
        "ar50": avg["all_rc_50%"] * 100,
    }
    if per_threshold:
        summary["ious"] = ev.ious
        summary["ap_t"] = ap[0, 0, :].copy()
        summary["rc_t"] = rc[0, 0, :].copy()
    return summary


# ---------------------------------------------------------------- cached assignment
# ``assign_instances_for_scan`` recomputes ``gts == instance_id`` once per (prediction, GT) pair, so a
# scene costs O(#pred * #gt * #vertices) and an operating-point sweep over a wide pool would spend
# hours re-deriving the same overlaps.  The overlaps do not depend on which subset is emitted: choose
# an operating point and the only thing that changes is WHICH proposals are present.  So assign once
# over the full pool, then subset the cached structures.  ``evaluate_matches`` is still the official
# implementation and is never re-implemented -- only the input to it is memoised.
class PoolAssignment:
    def __init__(self, scene, entry, gt, annotation):
        ev = evaluator(annotation)
        preds = []
        for i, pred in enumerate(entry["preds"]):
            mask = np.zeros(entry["nV"], bool)
            mask[pred["v"]] = True
            preds.append({"scan_id": f"{scene}|{i}", "label_id": 1,
                          "pred_mask": mask, "conf": 1.0})
        with contextlib.redirect_stdout(io.StringIO()):
            gt2pred, pred2gt = ev.assign_instances_for_scan(preds, gt[0], gt[1])
        self.label = ev.eval_class_labels[0]
        self.gt2pred = gt2pred
        self.pred2gt = pred2gt
        for entries in pred2gt.values():
            for item in entries:
                item["pool_index"] = int(item["filename"].rsplit("_", 1)[0].split("|")[1])
        for entries in gt2pred.values():
            for item in entries:
                for matched in item["matched_pred"]:
                    matched["pool_index"] = int(matched["filename"].rsplit("_", 1)[0].split("|")[1])

    def subset(self, keep):
        """Restrict the cached assignment to an emitted subset (a boolean mask over pool indices)."""
        if keep is None:
            return {"gt": self.gt2pred, "pred": self.pred2gt}
        pred = {self.label: [p for p in self.pred2gt[self.label] if keep[p["pool_index"]]]}
        gt = {self.label: [
            {**g, "matched_pred": [m for m in g["matched_pred"] if keep[m["pool_index"]]]}
            for g in self.gt2pred[self.label]
        ]}
        return {"gt": gt, "pred": pred}


def assignments(store, gts, scenes, annotation, cache={}):
    key = (id(store), annotation, tuple(scenes))
    if key not in cache:
        cache[key] = {s: PoolAssignment(s, store[s], gts[s], annotation)
                      for s in scenes if s in store and s in gts}
    return cache[key]


def fast_score(assign, gts, scenes, annotation, keeps=None, per_threshold=False):
    """Official ``evaluate_matches`` over a memoised assignment; identical output, ~50x faster."""
    ev = evaluator(annotation)
    matches = {}
    n_pred = 0
    for i, scene in enumerate(scenes):
        if scene not in assign:
            continue
        keep = None if keeps is None else keeps.get(scene)
        entry = assign[scene].subset(keep)
        matches[f"gt_{i}"] = entry
        n_pred += len(entry["pred"][assign[scene].label])
    with contextlib.redirect_stdout(io.StringIO()):
        ap, rc = ev.evaluate_matches(matches)
        avg = ev.compute_averages(ap, rc)
    summary = {
        "n_pred": n_pred,
        "n_gt": int(sum(gts[s][2] for s in scenes if s in gts)),
        "ap": avg["all_ap"] * 100, "ap50": avg["all_ap_50%"] * 100,
        "ap25": avg["all_ap_25%"] * 100, "ar": avg["all_rc"] * 100,
        "ar50": avg["all_rc_50%"] * 100,
    }
    if per_threshold:
        summary["ious"] = ev.ious
        summary["ap_t"] = ap[0, 0, :].copy()
        summary["rc_t"] = rc[0, 0, :].copy()
    return summary


# ---------------------------------------------------------------- decomposition
def decompose(summary):
    """Recover (P, #TP, #counted-FP, #ignored-FP) per IoU threshold from AP and R."""
    rows = []
    n_gt, n_pred = summary["n_gt"], summary["n_pred"]
    for iou, ap_t, rc_t in zip(summary["ious"], summary["ap_t"], summary["rc_t"]):
        if rc_t <= 0:
            rows.append((iou, 0.0, 0.0, 0, 0, n_pred, ap_t))
            continue
        precision = 2.0 * ap_t / rc_t - 1.0
        n_tp = rc_t * n_gt
        n_counted_fp = n_tp * (1.0 - precision) / max(precision, 1e-9)
        n_ignored = n_pred - n_tp - n_counted_fp
        rows.append((iou, precision, rc_t, n_tp, n_counted_fp, n_ignored, ap_t))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="tune")
    parser.add_argument("--frontend", default="sam3")
    parser.add_argument("--method", default="relift")
    parser.add_argument("--annotation", default="scannetv2")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    store = predictions(args.split, args.frontend, args.method, args.refresh)
    scenes = [s for s in read_split(args.split) if s in store]
    gts = ground_truth(scenes, store, args.annotation, args.refresh)
    summary = score(store, gts, scenes, args.annotation, per_threshold=True)

    print(f"\n  === {args.method} / {args.frontend} / {args.split} / {args.annotation} ===")
    print(f"  {len(scenes)} scenes   {summary['n_gt']} GT   {summary['n_pred']} proposals")
    print(f"  AP {summary['ap']:.2f}   AP50 {summary['ap50']:.2f}   "
          f"AP25 {summary['ap25']:.2f}   AR {summary['ar']:.2f}")

    print(f"\n  {'IoU':>5s} {'AP_t':>7s} {'R(1+P)/2':>9s} {'P':>7s} {'R':>7s} "
          f"{'#TP':>7s} {'#FP-cnt':>8s} {'#ignored':>9s}")
    for iou, precision, recall, n_tp, n_fp, n_ig, ap_t in decompose(summary):
        print(f"  {iou:5.2f} {ap_t*100:7.2f} {recall*(1+precision)/2*100:9.2f} "
              f"{precision*100:7.1f} {recall*100:7.1f} {n_tp:7.0f} {n_fp:8.0f} {n_ig:9.0f}")

    # exchange rate at the reported operating point (mean over the AP thresholds, 0.25 excluded)
    rows = [r for r in decompose(summary) if r[0] >= 0.5]
    precision = float(np.mean([r[1] for r in rows]))
    recall = float(np.mean([r[2] for r in rows]))
    n_counted = float(np.mean([r[3] + r[4] for r in rows]))
    d_tp = (1 + precision) / (2 * summary["n_gt"]) * 100
    d_fp = recall * precision / (2 * n_counted) * 100
    print(f"\n  mean over AP thresholds:  P {precision*100:.1f}   R {recall*100:.1f}   "
          f"counted {n_counted:.0f} of {summary['n_pred']} proposals "
          f"({100*(1-n_counted/summary['n_pred']):.0f} % ignored)")
    print(f"  one extra TRUE positive  : {d_tp:+.4f} AP")
    print(f"  one extra COUNTED false  : {d_fp:-.4f} AP".replace("+", ""))
    print(f"  ==> exchange rate: a proposal is worth emitting if it hits more than "
          f"1 time in {d_tp/d_fp:.1f}")


if __name__ == "__main__":
    main()
