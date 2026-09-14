#!/usr/bin/env python
"""Widen the hypothesis bank the joint objective may consider -- the one backend lever left.

WHY THIS IS NOT A SIXTH CANDIDATE GENERATOR.  The five rejected mechanisms all ADD predictions to a
finished partition, and all five died on the same arithmetic: constant-confidence AP needs a new
prediction to be right ~28% of the time and they are right 3-7% of the time.  This changes nothing
about what is emitted.  It changes what the joint stage is allowed to CONSIDER while re-partitioning
superpoints it already owns:

    bank, _ = e.evolve(e.seeds(frames), frames, one_step=True)
    _, _, scores = e.correspond(bank, frames)
    bank = bank[(scores >= .7) & ((bank @ e.total) >= 100)]      # c1_information_selected.py

That 0.7 correspondence floor is inherited, never fitted.  A hypothesis below it is discarded before
the variation-of-information objective ever scores it, so the objective is choosing from a bank
somebody else pruned.  Since the joint stage is measurably the strongest single component we have
(+1.683 AP / +0.862 AP50 / +0.323 AP25 over independent resolution of the same bank, CropFormer,
312 scenes), how much structure it is permitted to see is worth one experiment.

The prediction COUNT is free to move either way here, which is the tell that this is a different
lever: the objective may split an incumbent it can now explain better, or merge two it now sees are
one. Fitted on ScanNet TRAIN only.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

SNAP = "/home/saad/Documents/Codex/2026-09-05/yo/work/spacesculptor-cropformer-current-20260911"
B = "/home/saad/Desktop/spacesculptor_c1ap25"
R = "/home/saad/Desktop/spacesculptor_old"


def one(job):
    scene, frontend, rec_dir, scans, floor = job
    sys.path[:0] = [SNAP, B, R]
    import c1_deployed as dep
    dep.install(frontend)
    import c1_bridge_ablation as cba
    import c1_candidate as cc
    import official_eval as oe
    from c1_relift_refinement import refine_candidate
    from c1_absorb import build as absorb
    from c1_residual_discovery import discover
    from c1_crossview_attractors import Attractors
    from c1_information_sparse import SparseSceneInformation as SceneInformation
    import spp_deployed as SD

    if scans:
        oe.SCANS = scans
    cfg = dict(cba.FRONTENDS[frontend])
    cfg["recordings"] = Path(rec_dir)
    nV = len(oe._read_ply_xyz(f"{oe.SCANS}/{scene}/{scene}_vh_clean_2.ply"))
    item = cc._inference_item(scene, nV, cfg, Path(rec_dir), False)
    if item is None:
        return scene, None
    cand = cc.predict_candidate_item(item, cfg)
    preds, _ = refine_candidate(item, cfg, cand, cc.PROJECTION_TOP_K, cc.RELIFT_MATCH_FLOOR,
                               cc.RELIFT_VOTE_FLOOR, cc.TEMPORAL_PROJECTION_FLOOR, cc.FUSION_NMS)
    raw = {"preds": [{"v": np.flatnonzero(np.asarray(p["pred_mask"]).astype(bool)).astype(np.int32),
                      "conf": float(p.get("conf", 1.0))} for p in preds], "nV": nV}
    base = absorb({scene: raw}, [scene], {scene: item["superpoints"]},
                  SD.THETA, retention=SD.RETENTION)[scene]
    incumbent, _ = discover(item, base)

    # --- refine_scene, with the bank floor exposed instead of hard-coded at 0.7 -----------------
    e = Attractors(item["recording"], item["superpoints"])
    frames = np.arange(e.nF)
    bank, _ = e.evolve(e.seeds(frames), frames, one_step=True)
    _, _, scores = e.correspond(bank, frames)
    bank = bank[(scores >= floor) & ((bank @ e.total) >= 100)]
    owners = np.zeros(e.nS, np.int32)
    for k, p in enumerate(incumbent["preds"], 1):
        spg = np.unique(e.spp[p["v"]])
        if np.any(owners[spg]) or int(e.total[spg].sum()) != len(p["v"]):
            return scene, None
        owners[spg] = k
    energy = SceneInformation(e, anchored=True)
    final, _ = energy.optimize(owners, bank, mode="mean")
    masks = np.array([final == k for k in np.unique(final) if k > 0], bool).reshape(-1, e.nS)
    return scene, (nV, [np.flatnonzero(row[e.spp]).astype(np.int32) for row in masks], len(bank))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="v96")
    ap.add_argument("--rec-dir", default=f"{R}/runs/scannet_ap/v96_train")
    ap.add_argument("--scans", default=f"{R}/datasets/scannet_train/scans")
    ap.add_argument("--annotation", default="scannetv2")
    ap.add_argument("--floors", default="0.7,0.55,0.4,0.85")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--scenes-file", default="", help="restrict to this scene list, so two "
                                                       "recording banks can be compared on exactly "
                                                       "the same scenes")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N scenes, so the selection "
                                                       "split and the confirmation split are disjoint")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dump", default="", help="cache each floor's predictions here so one run can "
                                               "be scored under several protocols without re-running")
    a = ap.parse_args()

    sys.path[:0] = [SNAP, B, R]
    import official_eval as oe
    if a.scans:
        oe.SCANS = a.scans
    scenes = sorted(p.stem for p in Path(a.rec_dir).glob("*.pkl"))
    if a.scenes_file:
        want = {l.strip() for l in open(a.scenes_file) if l.strip()}
        scenes = [s for s in scenes if s in want]
    scenes = scenes[a.skip:]
    scenes = scenes[:a.limit or None]
    floors = [float(x) for x in a.floors.split(",")]

    res = {}
    for f in floors:
        got = {}
        with ProcessPoolExecutor(a.workers) as pool:
            futs = {pool.submit(one, (s, a.frontend, a.rec_dir, a.scans, f)): s for s in scenes}
            for fu in as_completed(futs):
                sc, r = fu.result()
                if r is not None:
                    got[sc] = r
        res[f] = got
        if a.dump:
            import pickle as _pk
            d = Path(a.dump) / f"floor{f}"
            d.mkdir(parents=True, exist_ok=True)
            for sc, (nV, vs, _) in got.items():
                with (d / f"{sc}.pkl").open("wb") as fh:
                    _pk.dump((nV, [(v, 1.0) for v in vs]), fh, protocol=4)
        print(f"    floor {f}: {len(got)} scenes, "
              f"{sum(len(v[1]) for v in got.values()):,} predictions, "
              f"bank median {int(np.median([v[2] for v in got.values()])):,}", flush=True)

    keep = sorted(set.intersection(*[set(v) for v in res.values()]))
    sems, inss, ngt = [], [], 0
    for sc in keep:
        nV = res[floors[0]][sc][0]
        if a.annotation == "scannetv2":
            sem, ins, k = oe.gt_labels(sc, nV, "scannetv2")
        elif a.annotation == "om3d":
            from c1_cdis_protocol import openmask3d_gt
            om = openmask3d_gt(sc, nV, "scannet200")
            ins = np.where(om >= 1000, (om % 1000) - 1, -1).astype(np.int64)
            u = np.unique(ins[ins >= 0])
            rm = {int(x): i for i, x in enumerate(u)}
            ins = np.array([rm.get(int(x), -1) for x in ins], np.int64)
            sem, k = np.where(ins >= 0, 2, 0).astype(np.int64), len(u)
        else:
            from scannet200_eval import gt200
            sem, ins, k = gt200(sc, nV)
        sems.append(sem)
        inss.append(ins)
        ngt += k

    sys.path.insert(0, "/home/saad/Desktop/spacesculptor_baselines/Open3DIS")
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval
    ev = ScanNetEval(class_labels=["object"], use_label=False,
                     dataset_name="scannet200" if a.annotation == "scannet200" else "scannetv2")
    Path("/tmp/claude-1001/jbank").mkdir(parents=True, exist_ok=True)
    print(f"\n  {len(keep)} scenes, {ngt:,} GT, {a.annotation}, front end {a.frontend}\n")
    print(f"  {'bank floor':12s} {'AP':>8s} {'AP50':>8s} {'AP25':>8s} {'rc@.25':>8s} {'preds':>7s}")
    base = None
    for f in floors:
        pl, npr = [], 0
        for i, sc in enumerate(keep):
            nV, vs, _ = res[f][sc]
            row = []
            for v in vs:
                v = v[(v >= 0) & (v < nV)]
                if len(v) < 100:
                    continue
                m = np.zeros(nV, bool)
                m[v] = True
                row.append({"scan_id": sc, "label_id": 1, "pred_mask": m, "conf": 1.0})
            npr += len(row)
            pl.append(row)
        with contextlib.redirect_stdout(io.StringIO()):
            r = ev.evaluate(pl, sems, inss, exp_path="/tmp/claude-1001/jbank")
        row3 = (r["all_ap"] * 100, r["all_ap_50%"] * 100, r["all_ap_25%"] * 100,
                r["all_rc_25%"] * 100)
        tag = " (deployed)" if f == 0.7 else ""
        if base is None and f == 0.7:
            base = row3
        d = "" if (base is None or f == 0.7) else (f"   ({row3[0]-base[0]:+.2f} / "
                                                   f"{row3[1]-base[1]:+.2f} / {row3[2]-base[2]:+.2f})")
        print(f"  {f:<12.2f} {row3[0]:8.3f} {row3[1]:8.3f} {row3[2]:8.3f} {row3[3]:8.3f} "
              f"{npr:7,d}{tag}{d}", flush=True)


if __name__ == "__main__":
    main()
