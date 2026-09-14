#!/usr/bin/env python
"""OFFICIAL ScanNet instance-segmentation evaluation (mask-AP), class-agnostic.

This replaces our own axis-aligned box-IoU scorer with the community-standard evaluator that
ScanNet/ScanNet200 papers use, so our numbers are directly comparable to published tables and cannot
be accused of a self-serving protocol. We use the evaluator shipped inside Open3DIS
(`open3dis/evaluation/scannetv2_inst_eval.py`, the standard ScanNet implementation) with
`use_label=False` -> CLASS-AGNOSTIC mask AP / AP50 / AP25, which is the correct setting for comparing
class-agnostic 3D instance proposals.  The published MV3DIS class-agnostic protocol sets every
evaluation confidence to 1.0.  This evaluator therefore defaults to that convention and exposes an
explicit ``--score-protocol method-native`` diagnostic for studying proposal ranking separately.

Every method here emits POINT MASKS (which all four natively produce -- the earlier box conversion
discarded information):
  ours      : linked cluster vertex ids (frozen v6 config, recomputed from the recorded masks)
  MC        : object_dict point_ids
  SAI3D     : official per-vertex 0/1 mask files
  Open3DIS  : official RLE point masks

  PYTHONPATH=<repo> <env>/python official_eval.py --scenes-file heldout_scenes.txt [--methods ...]
"""
from __future__ import annotations
import argparse, csv, glob, json, os, pickle, sys
import numpy as np
# NumPy 2.0 removed `np.in1d`; the ScanNet evaluator still calls it (scannetv2_inst_eval.py:310) on a
# 1-D array, where `np.isin` is its exact documented replacement. Shim it rather than editing the
# official evaluator, so the scoring code stays byte-identical to the published one.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
def _env(k, d):
    """Path overridable by environment so this package runs on any machine."""
    return _os.environ.get(k, d)
REPO = _env("C1_REPO", _os.path.dirname(_HERE))
BASE = _env("C1_BASE", _os.path.dirname(_HERE))
sys.path.insert(0, REPO); sys.path.insert(0, BASE)
sys.path.insert(0, f"{BASE}/Open3DIS")
from scripts.scannet_io import load_gt_instances, _read_ply_xyz
SCANS = _env("C1_SCANS", f"{REPO}/scans")
LABEL_MAP = _env("C1_LABEL_MAP", f"{REPO}/scannetv2-labels.combined.tsv")
_spp_cache = {}
MCROOT = f"{BASE}/MaskClustering/data/scannet/processed"
HELD = os.environ.get("HELD_DIR", f"{REPO}/runs/scannet_ap/v5_heldout")
# "selfiou" = deployed J-selection (RESULTS 2.4b); "adaptive" reproduces the pre-2.4b numbers.
SNAP_MODE = os.environ.get("SNAP_MODE", "selfiou")
NMS_SIZE_RATIO = float(os.environ.get("NMS_SIZE_RATIO", "0.3"))
CLAIM_MODE = os.environ.get("CLAIM_MODE", "frame")
GT_PROTOCOL = os.environ.get("GT_PROTOCOL", "scannetv2")
VETO_MODE = os.environ.get("VETO_MODE", "mean")
CONFLICT_FRAC = float(os.environ.get("CONFLICT_FRAC", "0.5"))
SCANNETV2_INSTANCE_IDS = frozenset((3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39))
_nyu40_by_raw = None


# ---------------- ground truth: per-point semantic + instance ids ----------------
def _load_nyu40_map():
    global _nyu40_by_raw
    if _nyu40_by_raw is None:
        with open(LABEL_MAP, newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            _nyu40_by_raw = {
                row["raw_category"].strip().lower(): int(row["nyu40id"])
                for row in rows
                if row.get("raw_category") and row.get("nyu40id")
            }
    return _nyu40_by_raw


def gt_labels(scene, nV, protocol=None):
    """Build class-agnostic ScanNetV2 GT from segmentation and aggregation metadata.

    ``scannetv2`` is the published class-agnostic benchmark protocol: only the 18 foreground NYU40
    classes are valid, and the official evaluator's 100-vertex minimum applies.  ``all-raw-legacy``
    reproduces this workspace's earlier non-standard protocol (every non-structural raw label and a
    200-vertex cutoff) so old numbers remain reproducible, but it must not be compared to published
    ScanNetV2 tables.

    The evaluator shifts semantic id 2 to its single valid class id 1, so all retained instances use
    semantic id 2 and background remains 0.
    """
    from scripts.scannet_io import STRUCTURAL
    protocol = protocol or GT_PROTOCOL
    if protocol not in {"scannetv2", "all-raw-legacy"}:
        raise ValueError(f"unknown GT protocol: {protocol}")
    nyu40 = _load_nyu40_map() if protocol == "scannetv2" else None
    min_vertices = 100 if protocol == "scannetv2" else 200
    segs = json.load(open(f"{SCANS}/{scene}/{scene}_vh_clean_2.0.010000.segs.json"))
    seg_idx = np.asarray(segs["segIndices"])[:nV]
    agg = json.load(open(f"{SCANS}/{scene}/{scene}.aggregation.json"))
    sem = np.zeros(nV, np.int64)
    ins = np.full(nV, -1, np.int64)
    k = 0
    for g in agg["segGroups"]:
        label = str(g["label"]).strip().lower()
        if protocol == "scannetv2":
            if nyu40.get(label) not in SCANNETV2_INSTANCE_IDS:
                continue
        elif label in STRUCTURAL:
            continue
        m = np.isin(seg_idx, np.asarray(g["segments"]))
        if m.sum() < min_vertices:
            continue
        sem[m] = 2
        ins[m] = k
        k += 1
    return sem, ins, k


# ---------------- per-method point-mask predictors ----------------
def _superpoints(scene, nV):
    # SPP_DIR lets a dataset that ships no over-segmentation supply one in the same format
    # (see mesh_oversegment.py). Set it to compare our re-derived segmentation against ScanNet's.
    d = os.environ.get("SPP_DIR")
    if d:
        f = f"{d}/{scene}.json"
        if os.path.exists(f):
            si = np.asarray(json.load(open(f))["segIndices"], np.int64)[:nV]
            if len(si) < nV:
                si = np.pad(si, (0, nV - len(si)), constant_values=-1)
            return np.unique(si, return_inverse=True)[1]
    segs = json.load(open(f"{SCANS}/{scene}/{scene}_vh_clean_2.0.010000.segs.json"))
    si = np.asarray(segs["segIndices"], np.int64)[:nV]
    if len(si) < nV:
        si = np.pad(si, (0, nV - len(si)), constant_values=-1)
    _, inv = np.unique(si, return_inverse=True)
    return inv


def _snap_selfiou(m, spp, spp_tot, ladder=(0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05),
                  min_pts=40):
    """Choose the snap threshold by AGREEMENT WITH THE CARVE, not by survival.

    The rule this replaces (`_snap_adaptive`) relaxes phi until the instance retains half its points --
    a test of whether the object survives, never of which threshold yields the best mask. An oracle that
    picks the best phi per instance scores 47.6 AP against 37.0 on DEV-16, so the choice is worth more
    than any other single decision in the component.

    There are no labels available, but the carved cloud C is evidence: a good snap should EXPLAIN it,
    absorbing the superpoints C fills and rejecting those it merely grazes. So maximise

        J(phi) = |M_phi ^ C| / |M_phi u C|

    over the ladder. J has no free parameters -- it is a closed-form argmax -- so there is nothing to
    overfit, and it references only the carve and the over-segmentation, never a property of the 2D
    segmenter (verified front-end agnostic: +0.9 AP on SAM 3, +0.7 on Grounded-SAM).
    """
    cnt = np.bincount(spp, weights=m.astype(np.float64), minlength=len(spp_tot))
    cov = cnt / np.maximum(spp_tot, 1)
    nC = float(m.sum())
    best = None
    for phi in ladder:
        K = cov >= phi
        if not K.any():
            continue
        inter = float(cnt[K].sum()); msize = float(spp_tot[K].sum())
        if msize < min_pts:
            continue
        J = inter / max(msize + nC - inter, 1e-9)
        if best is None or J > best[0]:
            best = (J, K)
    if best is None:
        return m
    return best[1][spp].astype(np.uint8)


def _snap_adaptive(m, spp, frac=0.5, min_keep=0.5):
    """Snap, but NEVER let snapping destroy an instance. Small objects often occupy a minority of
    each superpoint they touch, so a fixed 50% rule erases them (measured: mask-R@.25 0.874 -> 0.727).
    We therefore relax the threshold for that instance until it retains >= `min_keep` of its points,
    and fall back to the raw mask if even the loosest rule cannot."""
    for f in (frac, 0.35, 0.2):
        ms = _snap_to_spp(m, spp, f)
        if ms.sum() >= min_keep * m.sum() and ms.sum() >= 40:
            return ms
    return m


def _mask_nms(preds, thr=0.7, size_ratio=0.3):
    """Duplicate suppression that distinguishes a DUPLICATE from a PART.

    A pure containment test (`inter / min(|a|,|b|) > thr`) cannot tell these apart:
      DUPLICATE  two proposals of one object, of comparable size        -> suppress
      PART       a small object nested in a larger, OVER-MERGED cluster -> must be kept
    For a part the ratio is ~1.0 by construction, so the old rule deleted every one of them and kept the
    over-merged parent -- the failure the clutter tier can least afford, and measurably worse on the
    finer ScanNet200 annotation than on ScanNetV2.

    The guard adds the missing condition: suppress only when the two are also of COMPARABLE SIZE, which
    a genuine duplicate is and a part is not. `size_ratio=0` recovers the previous behaviour.
    Measured (TUNE-8 / DEV-16, J-selection held fixed): ScanNetV2 +0.3/+0.4 AP, ScanNet200 +0.3/+0.5 AP.
    Removing suppression altogether is worse than either (40.8 V2 / 34.9 S200 on TUNE), so the rule is
    load-bearing and only its criterion was wrong.
    """
    preds = sorted(preds, key=lambda p: -p["conf"])
    kept = []
    for p in preds:
        a = p["pred_mask"].astype(bool); sa = int(a.sum())
        dup = False
        for q in kept:
            b = q["pred_mask"].astype(bool); sb = int(b.sum())
            inter = int(np.logical_and(a, b).sum())
            if inter / max(min(sa, sb), 1) <= thr:
                continue
            if min(sa, sb) / max(max(sa, sb), 1) > size_ratio:   # comparable size -> a real duplicate
                dup = True; break
        if not dup:
            kept.append(p)
    return kept


def _snap_to_spp(m, spp, frac=0.5):
    """Snap a ragged point mask to ScanNet's geometric over-segmentation: a superpoint joins the
    instance iff a fraction >= `frac` of its vertices are in the mask. Our clusters are unions of
    projected 2D-mask votes and are therefore ragged at object boundaries, while the mesh
    over-segmentation follows true geometric edges -- the same primitives SAI3D/Open3DIS/MC use.
    This makes the predicted MASK as clean as the predicted BOX (mask-IoU is far less forgiving
    than box-IoU of boundary noise)."""
    nspp = spp.max() + 1
    cnt = np.bincount(spp, weights=m.astype(np.float64), minlength=nspp)
    tot = np.bincount(spp, minlength=nspp).astype(np.float64)
    keep = (cnt / np.maximum(tot, 1)) >= frac
    return keep[spp].astype(np.uint8)


def _spp_purity(vid, spp, spp_tot):
    """Fraction of a cluster's vertices lying in superpoints it occupies almost WHOLLY (>=80 %).

    Mask AP is a RANKED metric, so a proposal's score has to predict its mask quality, not merely its
    existence. Multi-view support alone cannot: a spurious blob glued together by a few co-occurring
    detections can be seen in many frames. Geometric coherence is the missing, complementary signal --
    a genuine object is a union of whole geometric primitives, whereas a spurious cluster slices
    across them, because the mesh over-segmentation follows real surface discontinuities.

    Measured on the 8 tuning scenes this correlates with true mask IoU at Spearman +0.514 (mean 0.468
    for clusters reaching IoU>=0.5 vs 0.146 for those that do not) -- nearly as strong as multi-view
    support (+0.544) and largely independent of it. The over-segmentation is the same GT-free primitive
    SAI3D, Open3DIS and MaskClustering all consume, so using it costs no supervision and no fairness.
    """
    cnt = np.bincount(spp[vid], minlength=len(spp_tot)).astype(np.float64)
    whole = (cnt / np.maximum(spp_tot, 1)) >= 0.8
    return float(cnt[whole].sum() / max(cnt.sum(), 1))


def preds_ours(scene, nV, snap=True, snap_frac=0.5, nms=0.5):   # DEPLOYED: snap + mask-NMS
    from v6_cluster import make_preds, link_masks
    from heldout_eval import CFG
    f = f"{HELD}/{scene}.pkl"
    if not os.path.exists(f):
        return []
    d = pickle.load(open(f, "rb"))
    sp = _spp_cache.get(scene)
    if sp is None:
        sp = _superpoints(scene, nV); _spp_cache[scene] = sp
    spp_tot = np.bincount(sp, minlength=sp.max() + 1).astype(np.float64)
    out = []
    for vids, nmask, nviews, mcons in link_masks(d, CFG["tau_same"], CFG["tau_cross"], CFG["veto"],
                                                 use_veto=CFG["use_veto"], carve=CFG["carve"],
                                                 greedy=CFG["greedy"], claim_mode=CLAIM_MODE,
                                                 veto_mode=VETO_MODE, conflict_frac=CONFLICT_FRAC):
        if len(vids) < 40:
            continue
        vid = vids[vids < nV]
        m = np.zeros(nV, np.uint8); m[vid] = 1
        if snap:
            ms = _snap_selfiou(m, sp, spp_tot) if SNAP_MODE == "selfiou" \
                else _snap_adaptive(m, sp, snap_frac)
            if ms.sum() >= 40:
                m = ms
        # MASK-QUALITY SCORE: multi-view support x geometric coherence, cubed.
        #
        # The exponent was chosen on the DEV-16 validation split, which shows a clear INTERIOR optimum
        # (AP 35.2 / 36.3 / 37.0 / 36.8 / 36.7 / 36.9 for k = 1..6) -- i.e. k=3 is a real maximum, not
        # a monotone trend cut off arbitrarily. TUNE-8 agrees that k>=3 beats k<=2. TEST was never
        # consulted. Both factors are load-bearing: coherence alone scores 36.3 and multi-view support
        # alone 31.4 on DEV, against 37.0 together. Scale-free rank fusion of the two is strictly worse
        # (32.5) -- their dynamic range carries information that ranking away destroys. Soft-NMS buys
        # +0.1 AP but costs 1.3 AP25, so it is not used.
        views = float(nviews) * (1.0 - np.exp(-len(vid) / 200.0))
        conf = views * _spp_purity(vid, sp, spp_tot) ** 3
        out.append({"scan_id": scene, "label_id": 1, "pred_mask": m, "conf": conf})
    mx = max([o["conf"] for o in out], default=1.0)
    for o in out:
        o["conf"] = o["conf"] / mx
    if nms > 0:
        out = _mask_nms(out, nms, NMS_SIZE_RATIO)
    return out


def preds_mc(scene, nV):
    # MaskClustering names the output subdirectory after its config, so it is `scannet` for ScanNet
    # and `replica` for Replica. Glob rather than hard-code, so the same evaluator serves both.
    f = f"{MCROOT}/{scene}/output/object/scannet/object_dict.npy"
    if not os.path.exists(f):
        cand = sorted(glob.glob(f"{MCROOT}/{scene}/output/object/*/object_dict.npy"))
        if not cand:
            return []
        f = cand[0]
    d = np.load(f, allow_pickle=True).item()
    out = []
    for v in d.values():
        pid = np.asarray(list(v["point_ids"]), int)
        pid = pid[pid < nV]
        if len(pid) < 50:
            continue
        m = np.zeros(nV, np.uint8); m[pid] = 1
        # MaskClustering's object_dict has no prediction-confidence field.  Its native class-agnostic
        # output is therefore unranked.  Do not manufacture a size score: doing so changes AP while
        # leaving the masks fixed and is neither the baseline's protocol nor MV3DIS's protocol.
        out.append({"scan_id": scene, "label_id": 1, "pred_mask": m, "conf": 1.0})
    return out


def preds_sai3d(scene, nV, eval_dir=f"{BASE}/SAI3D/data/ScanNet/results/run"):
    idx = f"{eval_dir}/{scene}.txt"
    if not os.path.exists(idx):
        return []
    out = []
    for line in open(idx):
        p = line.split()
        if len(p) < 3:
            continue
        mf = os.path.join(eval_dir, p[0])
        if not os.path.exists(mf):
            continue
        m = np.loadtxt(mf, dtype=np.int32)
        if len(m) < nV:
            m = np.pad(m, (0, nV - len(m)))
        m = (m[:nV] > 0).astype(np.uint8)
        if m.sum() < 40:
            continue
        # The third column is SAI3D's exported confidence (1.0 in its official class-agnostic files).
        # Earlier versions of this evaluator ignored it and invented a mask-size score, which was a
        # protocol bug rather than a property of SAI3D.
        out.append({"scan_id": scene, "label_id": 1, "pred_mask": m, "conf": float(p[2])})
    return out


def _rle_decode(rle):
    length = rle["length"]; s = rle["counts"]
    starts, nums = [np.asarray(x, np.int64) for x in (s[0:][::2], s[1:][::2])]
    starts = starts - 1; ends = starts + nums
    m = np.zeros(length, np.uint8)
    for lo, hi in zip(starts, ends):
        m[lo:hi] = 1
    return m


def preds_open3dis(scene, nV, src=f"{BASE}/open3dis_exp/ours_masks/hier_agglo"):
    import torch
    f = f"{src}/{scene}.pth"
    if not os.path.exists(f):
        return []
    d = torch.load(f, weights_only=False)
    conf = d["conf"].cpu().numpy() if hasattr(d["conf"], "cpu") else np.asarray(d["conf"])
    out = []
    for k, r in enumerate(d["ins"]):
        m = _rle_decode(r)
        if len(m) < nV:
            m = np.pad(m, (0, nV - len(m)))
        m = m[:nV]
        if m.sum() < 40:
            continue
        out.append({"scan_id": scene, "label_id": 1, "pred_mask": m, "conf": float(conf[k])})
    return out


def preds_any3dis(scene, nV, src=f"{BASE}/any3dis_exp/version_dp_maximum_score_0.6_n_spp_div4/mask2d_lifted"):
    """Any3DIS (CVPR'25) official-repo output: RLE point masks from its 3D mask-optimisation stage.
    NOTE Any3DIS runs its OWN 2D front-end (SAM-2 tracking) by design, so unlike MC/SAI3D/Open3DIS it
    does not consume our SAM3 masks -- it belongs in a system-vs-system table, not the identical-mask one."""
    import torch
    f = f"{src}/{scene}.pth"
    if not os.path.exists(f):
        return []
    d = torch.load(f, weights_only=False)
    out = []
    source_conf = d.get("conf")
    if source_conf is not None:
        source_conf = source_conf.cpu().numpy() if hasattr(source_conf, "cpu") else np.asarray(source_conf)
    for k, r in enumerate(d["ins"]):
        m = _rle_decode(r)
        if len(m) < nV:
            m = np.pad(m, (0, nV - len(m)))
        m = m[:nV]
        if m.sum() < 40:
            continue
        # Use a confidence only when the official output actually supplies one.  Otherwise the native
        # class-agnostic result is unranked, just like the protocol used in the Any3DIS paper.
        conf = float(source_conf[k]) if source_conf is not None else 1.0
        out.append({"scan_id": scene, "label_id": 1, "pred_mask": m, "conf": conf})
    return out


def _apply_score_protocol(preds, protocol):
    """Apply an evaluation-only score convention without changing masks or proposal order."""
    if protocol == "method-native":
        return preds
    if protocol != "constant":
        raise ValueError(f"unknown score protocol: {protocol}")
    out = []
    for pred in preds:
        item = pred.copy()
        item["conf"] = 1.0
        out.append(item)
    return out


def preds_candidate(scene, nV):
    """Frozen temporal/projection fusion; imported lazily to avoid evaluator import cycles."""
    from c1_candidate import predict_from_environment
    return predict_from_environment(scene, nV)


def preds_candidate_baseline(scene, nV):
    """Frontend-matched deployed path with the validated 0.003 export floor."""
    from c1_candidate import deployed_from_environment
    return deployed_from_environment(scene, nV)


def preds_candidate_relift(scene, nV):
    """Frozen proposal-conditioned multi-view re-lifting candidate."""
    from c1_candidate import relift_from_environment
    return relift_from_environment(scene, nV)


def _evaluate_table(store, gts, scenes, methods, annotation):
    """Evaluate an already-generated proposal store against one annotation mapping."""
    from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval
    evaluator = ScanNetEval(
        class_labels=["object"], use_label=False,
        dataset_name="scannet200" if annotation == "scannet200" else "scannetv2",
    )
    n_gt = sum(gts[scene][2] for scene in scenes)
    print(
        f"\n  === OFFICIAL {annotation} mask-AP (class-agnostic), "
        f"{len(scenes)} scenes, {n_gt} GT ==="
    )
    print(f"  {'method':22s} {'#pred':>6s} {'AP':>7s} {'AP50':>7s} {'AP25':>7s} {'AR':>7s}")
    results = {}
    for method in methods:
        pred_list = [store[method][scene] for scene in scenes]
        sem_list = [gts[scene][0] for scene in scenes]
        ins_list = [gts[scene][1] for scene in scenes]
        try:
            average = evaluator.evaluate(pred_list, sem_list, ins_list, exp_path="/tmp")
            n_predictions = sum(len(predictions) for predictions in pred_list)
            summary = {
                "predictions": int(n_predictions),
                "ap": float(average["all_ap"] * 100.0),
                "ap50": float(average["all_ap_50%"] * 100.0),
                "ap25": float(average["all_ap_25%"] * 100.0),
                "ar": float(average["all_rc"] * 100.0),
                "ar50": float(average["all_rc_50%"] * 100.0),
                "ar25": float(average["all_rc_25%"] * 100.0),
            }
            results[method] = summary
            print(
                f"  {NAMES[method]:22s} {n_predictions:6d} {summary['ap']:7.1f} "
                f"{summary['ap50']:7.1f} {summary['ap25']:7.1f} {summary['ar']:7.1f}"
            )
        except Exception as error:
            results[method] = {"error": f"{type(error).__name__}: {error}"}
            print(f"  {NAMES[method]:22s}  FAILED: {type(error).__name__}: {error}")
    return {"scenes": len(scenes), "gt": int(n_gt), "methods": results}


METHODS = {"ours": preds_ours, "ours_candidate": preds_candidate,
           "ours_candidate_relift": preds_candidate_relift,
           "ours_candidate_baseline": preds_candidate_baseline,
           "any3dis": preds_any3dis, "ours_nosnap": lambda sc, nV: preds_ours(sc, nV, snap=False),
           "ours_nms7": lambda sc, nV: preds_ours(sc, nV, nms=0.7),
           "ours_nms5": lambda sc, nV: preds_ours(sc, nV, nms=0.5), "mc": preds_mc, "sai3d": preds_sai3d, "open3dis": preds_open3dis}
NAMES = {"ours": "OURS (deployed)", "ours_candidate": "OURS temporal fusion",
         "ours_candidate_relift": "OURS proposal re-lift",
         "ours_candidate_baseline": "OURS matched baseline",
         "any3dis": "Any3DIS (own SAM-2)", "ours_nosnap": "OURS (raw masks)",
         "ours_nms7": "OURS +snap +maskNMS.7", "ours_nms5": "OURS +snap +maskNMS.5", "mc": "MaskClustering", "sai3d": "SAI3D", "open3dis": "Open3DIS (2D)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-file", default=f"{BASE}/heldout_scenes.txt")
    ap.add_argument("--methods", default="mc,sai3d,open3dis,ours")
    ap.add_argument(
        "--score-protocol",
        choices=("constant", "method-native"),
        default="constant",
        help=("evaluation confidence convention; 'constant' matches the published MV3DIS "
              "class-agnostic protocol, while 'method-native' audits each exported score"),
    )
    ap.add_argument(
        "--gt-protocol",
        choices=("scannetv2", "all-raw-legacy"),
        default=GT_PROTOCOL,
        help="published 18-class ScanNetV2 GT, or the earlier non-standard all-raw diagnostic",
    )
    ap.add_argument("--only-common", action="store_true",
                    help="restrict to scenes every requested method produced output for")
    ap.add_argument("--dump", default=None, help="save per-scene per-method masks for bootstrap")
    ap.add_argument("--also-scannet200", action="store_true",
                    help="reuse the generated proposals for a second ScanNet200 annotation evaluation")
    ap.add_argument("--json-out", default=None, help="save metric summaries and protocol metadata")
    ap.add_argument(
        "--allow-legacy-oracle-recordings",
        action="store_true",
        help=("explicitly allow pre-audit candidate recordings whose prompts came from per-scene "
              "validation annotations; never use this for a clean end-to-end claim"),
    )
    args = ap.parse_args()
    scenes = [s.strip() for s in open(args.scenes_file) if s.strip()]
    methods = args.methods.split(",")
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        ap.error(f"unknown methods: {', '.join(unknown_methods)}")
    candidate_requested = any(method.startswith("ours_candidate") for method in methods)
    if args.allow_legacy_oracle_recordings and not candidate_requested:
        ap.error("--allow-legacy-oracle-recordings only applies to ours_candidate methods")
    if candidate_requested:
        # Candidate evaluation fails closed.  Cached oracle recordings remain available for explicitly
        # labelled backend ablations, but an omitted flag can never silently turn one into a clean run.
        os.environ["C1_REQUIRE_CLEAN_PROMPTS"] = (
            "0" if args.allow_legacy_oracle_recordings else "1"
        )

    store = {m: {} for m in methods}
    gts = {}
    gts_scannet200 = {}
    for sc in scenes:
        nV = len(_read_ply_xyz(f"{SCANS}/{sc}/{sc}_vh_clean_2.ply"))
        sem, ins, ngt = gt_labels(sc, nV, args.gt_protocol)
        gts[sc] = (sem, ins, ngt)
        if args.also_scannet200:
            from scannet200_eval import gt200
            mapped = gt200(sc, nV)
            if mapped is not None:
                gts_scannet200[sc] = mapped
        for m in methods:
            store[m][sc] = _apply_score_protocol(METHODS[m](sc, nV), args.score_protocol)
    if args.only_common:
        keep = [sc for sc in scenes if all(len(store[m][sc]) > 0 for m in methods)]
        print(f"  [common-scene mode] {len(keep)}/{len(scenes)} scenes")
        scenes = keep
    print("\n  === OFFICIAL class-agnostic evaluation ===")
    print(f"  score protocol: {args.score_protocol}")
    print(f"  GT protocol: {args.gt_protocol}")
    summaries = {
        "scannetv2": _evaluate_table(store, gts, scenes, methods, "scannetv2")
    }
    if args.also_scannet200:
        scenes200 = [scene for scene in scenes if scene in gts_scannet200]
        summaries["scannet200"] = _evaluate_table(
            store, gts_scannet200, scenes200, methods, "scannet200"
        )
    if args.dump:
        pickle.dump(
            {
                "store": store,
                "gts": gts,
                "gts_scannet200": gts_scannet200,
                "scenes": scenes,
            },
            open(args.dump, "wb"),
        )
        print(f"  [dumped] {args.dump}")
    if args.json_out:
        payload = {
            "scenes_file": os.path.abspath(args.scenes_file),
            "score_protocol": args.score_protocol,
            "gt_protocol": args.gt_protocol,
            "frontend": os.environ.get("C1_FRONTEND"),
            "recordings": os.environ.get("HELD_DIR", HELD),
            "require_clean_prompts": os.environ.get("C1_REQUIRE_CLEAN_PROMPTS", "1") == "1",
            "recording_protocol": (
                "legacy-scene-oracle"
                if args.allow_legacy_oracle_recordings
                else "clean-fixed-or-prompt-free"
            ),
            "summaries": summaries,
        }
        with open(args.json_out, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"  [wrote] {args.json_out}")


if __name__ == "__main__":
    main()
