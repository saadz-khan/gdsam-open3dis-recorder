#!/usr/bin/env python
"""The deployed pipeline on ScanNet++ v1 validation, class-agnostic instance segmentation.

The backend is IDENTICAL to the ScanNet path -- same constants, same coarse prior, same mass
retention, same joint inference -- so a difference between the two datasets is a property of the
data, not of the configuration. Three things necessarily differ, and each is explicit:

  * recordings come from `v5_record_spp` (iPhone RGB-D, aligned_pose, 640x480 working grid);
  * superpoints are OURS, because ScanNet++ ships no over-segmentation;
  * ground truth is the 83-class benchmark instance list, `SPLIT` and structure excluded.

  PYTHONPATH=/home/saad/Desktop/spacesculptor_old <sp311>/python spp_deployed.py \
      --recordings runs/scannetpp/sam3_val --joint
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np

REPO = Path("/home/saad/Desktop/spacesculptor_old")
BASE = Path(__file__).resolve().parent
SHARED = "/home/saad/Desktop/spacesculptor_baselines"


def _own_the_path() -> None:
    sys.path[:] = [q for q in sys.path if not str(q).startswith(SHARED)]
    for q in (REPO, BASE, BASE / "Open3DIS"):
        r = str(q)
        while r in sys.path:
            sys.path.remove(r)
        sys.path.insert(0, r)


_own_the_path()
if not hasattr(np, "in1d"):
    np.in1d = np.isin

THETA, RETENTION = 0.65, "mass"


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--recordings", required=True)
    ap_.add_argument("--discovery", action="store_true")
    ap_.add_argument("--joint", action="store_true")
    ap_.add_argument("--limit", type=int, default=0)
    ap_.add_argument("--all-instances", action="store_true",
                     help="score every annotated instance instead of the 83 benchmark classes; "
                          "a diagnostic, not the standard protocol")
    args = ap_.parse_args()

    import c1_deployed as D
    w = D.install("v96")                       # coarse prior, asserted -- same as ScanNet
    import c1_bridge_ablation as cba, c1_candidate as cc, c1_apdecomp as ap
    import scannetpp_eval as E
    from c1_relift_refinement import refine_candidate
    from c1_absorb import build as absorb

    rec = Path(args.recordings)
    scenes = sorted(p.stem for p in rec.glob("*.pkl"))
    if args.limit:
        scenes = scenes[:args.limit]
    if not scenes:
        raise SystemExit(f"no recordings in {rec}")
    cfg = dict(cba.FRONTENDS["v96"])
    print(f"  {len(scenes)} scenes, coarse prior {w:.2f}, theta {THETA}, retention {RETENTION}",
          flush=True)

    store, spps, t0 = {}, {}, time.time()
    for i, s in enumerate(scenes, 1):
        item = E.spp_item(s, rec)
        if item is None:
            continue
        cand = cc.predict_candidate_item(item, cfg)
        preds, _ = refine_candidate(item, cfg, cand, cc.PROJECTION_TOP_K, cc.RELIFT_MATCH_FLOOR,
                                    cc.RELIFT_VOTE_FLOOR, cc.TEMPORAL_PROJECTION_FLOOR,
                                    cc.FUSION_NMS)
        store[s] = {"preds": [{"v": np.flatnonzero(np.asarray(p["pred_mask"]).astype(bool)
                                                   ).astype(np.int32),
                               "conf": float(p.get("conf", 1.0))} for p in preds],
                    "nV": item["n_vertices"]}
        spps[s] = item["superpoints"]
        if i % 10 == 0:
            print(f"    {i}/{len(scenes)}  ({time.time()-t0:.0f}s)", flush=True)

    used = sorted(store)
    result = absorb(store, used, spps, THETA, retention=RETENTION)

    if args.discovery or args.joint:
        from c1_residual_discovery import discover
        for s in used:
            item = E.spp_item(s, rec)
            result[s], _ = discover(item, result[s])
            if args.joint:
                from c1_information_selected import refine_scene
                result[s], _ = refine_scene(item, result[s])

    n = sum(len(v["preds"]) for v in result.values())
    print(f"  {n} proposals over {len(used)} scenes\n", flush=True)

    classes = None
    if args.all_instances:                     # diagnostic variant
        import json
        classes = set()
        for s in used:
            for g in json.load(open(E.SPP / s / "scans/segments_anno.json"))["segGroups"]:
                lab = str(g.get("label", "")).strip()
                if lab and lab != "SPLIT":
                    classes.add(lab)

    gts = {}
    for s in used:                             # GT is opened only now, after predictions are fixed
        g = E.gt_scannetpp(s, result[s]["nV"], classes)
        if g is not None and g[2] > 0:
            gts[s] = g
    ok = [s for s in used if s in gts]
    r = ap.fast_score(ap.assignments(result, gts, ok, "scannetv2"), gts, ok, "scannetv2")
    tag = "all annotated instances" if args.all_instances else "83 benchmark classes"
    print(f"  ScanNet++ v1 val ({tag}), {len(ok)} scenes", flush=True)
    print(f"    AP {r['ap']:6.2f}   AP50 {r['ap50']:6.2f}   AP25 {r['ap25']:6.2f}"
          f"   n_gt={int(r['n_gt'])} n_pred={int(r['n_pred'])}", flush=True)


if __name__ == "__main__":
    main()
