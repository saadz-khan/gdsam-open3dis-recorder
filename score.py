#!/usr/bin/env python
"""Score one or more Grounded-SAM recording banks with the deployed C1 backend.

Runs the full pipeline per scene -- relift, retention absorb, residual discovery, anchored
variation-of-information joint inference at hypothesis-bank floor 0.55 -- freezes every prediction,
and only then opens ground truth.  Scoring is the untouched official class-agnostic ScanNet
evaluator at constant export confidence 1.0.

Scores whatever scenes a bank actually contains, so it can be pointed at a run that is still in
progress.  With several banks it restricts to the scenes they share, which is what makes the
comparison matched.

Reported per bank:
  masks/scene   recorded 2D masks lifted into the scene, a front-end property
  reach@.25     share of eligible GT that ANY single recorded mask already covers at IoU .25 --
                an upper bound no backend can exceed, and the stable signal on small samples
  AP/AP50/AP25  the official metric; set-level, so it needs a sizeable scene count to be read
  rc@.25        the evaluator's own recall
  preds         predictions surviving the evaluator's 100-vertex floor

  python score.py --banks ours=/path/a openis=/path/b --scans /data/scannet/scans \
                  --label-map /data/scannetv2-labels.combined.tsv --annotation scannetv2
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BANK_FLOOR = 0.55


def _paths(a):
    os.environ["C1_SCANS"] = str(Path(a.scans).resolve())
    if a.label_map:
        os.environ["C1_LABEL_MAP"] = str(Path(a.label_map).resolve())
    if a.s200:
        os.environ["C1_S200"] = str(Path(a.s200).resolve())
    os.environ.setdefault("C1_REPO", str(HERE))
    for p in (str(HERE / "backend"), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def predict(job):
    """Deployed prediction for one scene: the frozen vertex sets the evaluator will score."""
    scene, rec_dir, env = job
    os.environ.update(env)
    for p in (str(HERE / "backend"), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    import c1_deployed as dep
    import c1_bridge_ablation as cba
    dep.install("gdsam")
    import c1_candidate as cc
    import official_eval as oe
    from c1_relift_refinement import refine_candidate
    from c1_absorb import build as absorb
    from c1_residual_discovery import discover
    from c1_crossview_attractors import Attractors
    from c1_information_sparse import SparseSceneInformation as SceneInformation
    import spp_deployed as SD

    cfg = dict(cba.FRONTENDS["gdsam"])
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
    inc, _ = discover(item, base)
    e = Attractors(item["recording"], item["superpoints"])
    frames = np.arange(e.nF)
    bank, _ = e.evolve(e.seeds(frames), frames, one_step=True)
    _, _, sc = e.correspond(bank, frames)
    bank = bank[(sc >= BANK_FLOOR) & ((bank @ e.total) >= 100)]
    owners = np.zeros(e.nS, np.int32)
    for k, p in enumerate(inc["preds"], 1):
        spg = np.unique(e.spp[p["v"]])
        if np.any(owners[spg]) or int(e.total[spg].sum()) != len(p["v"]):
            return scene, None
        owners[spg] = k
    final, _ = SceneInformation(e, anchored=True).optimize(owners, bank, mode="mean")
    masks = np.array([final == k for k in np.unique(final) if k > 0], bool).reshape(-1, e.nS)
    return scene, (nV, [np.flatnonzero(r[e.spp]).astype(np.int32) for r in masks])


def gt_for(scene, nV, annotation, oe):
    if annotation == "scannetv2":
        return oe.gt_labels(scene, nV, "scannetv2")
    from scannet200_eval import gt200
    return gt200(scene, nV)


def reach(rec_dir, scene, annotation, oe, T=0.25):
    rec = pickle.load(open(Path(rec_dir) / f"{scene}.pkl", "rb"))
    nV = rec["P"].shape[0]
    Pc = rec["P"].tocsc()
    got = gt_for(scene, nV, annotation, oe)
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
            i = int(gs[vv].sum())
            if i and i / (len(gm) + len(vv) - i) >= T:
                hit += 1
                break
    return hit, tot, rec["nM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", required=True, help="name=/path/to/recordings ...")
    ap.add_argument("--scans", required=True)
    ap.add_argument("--label-map", default="")
    ap.add_argument("--s200", default="")
    ap.add_argument("--annotation", default="scannetv2", choices=("scannetv2", "scannet200"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    _paths(a)
    env = {k: os.environ[k] for k in ("C1_SCANS", "C1_LABEL_MAP", "C1_S200", "C1_REPO")
           if k in os.environ}

    banks = []
    for spec in a.banks:
        name, _, path = spec.partition("=")
        banks.append((name or Path(path).name, path))
    common = None
    for _, d in banks:
        s = {p.stem for p in Path(d).glob("*.pkl")}
        common = s if common is None else (common & s)
    scenes = sorted(common)[:a.limit or None]
    if not scenes:
        print("  no scene present in every bank yet")
        return
    print(f"  {len(scenes)} scene(s) in all {len(banks)} bank(s), {a.annotation}, "
          f"bank floor {BANK_FLOOR}, constant confidence 1.0\n", flush=True)

    import official_eval as oe
    sys.path.insert(0, str(HERE))
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval
    ev = ScanNetEval(class_labels=["object"], use_label=False,
                     dataset_name="scannet200" if a.annotation == "scannet200" else "scannetv2")
    out = Path("/tmp/c1_score_out")
    out.mkdir(parents=True, exist_ok=True)

    print(f"  {'bank':28s} {'masks/sc':>9s} {'reach@.25':>10s} {'AP':>8s} {'AP50':>8s} "
          f"{'AP25':>8s} {'rc@.25':>8s} {'preds':>7s}")
    for name, d in banks:
        h = t = m = 0
        for sc in scenes:
            hh, tt, mm = reach(d, sc, a.annotation, oe)
            h += hh
            t += tt
            m += mm
        got = {}
        with ProcessPoolExecutor(a.workers) as pool:
            futs = {pool.submit(predict, (sc, d, env)): sc for sc in scenes}
            for fu in as_completed(futs):
                s_, r = fu.result()
                if r is not None:
                    got[s_] = r
        keep = [s for s in scenes if s in got]
        sems, inss = [], []
        for sc in keep:
            nV = got[sc][0]
            sem, ins, _ = gt_for(sc, nV, a.annotation, oe)
            sems.append(sem)
            inss.append(ins)
        pl, npr = [], 0
        for i, sc in enumerate(keep):
            nV, vs = got[sc]
            row = []
            for v in vs:
                v = v[(v >= 0) & (v < nV)]
                if len(v) < 100:
                    continue
                msk = np.zeros(nV, bool)
                msk[v] = True
                row.append({"scan_id": sc, "label_id": 1, "pred_mask": msk, "conf": 1.0})
            npr += len(row)
            pl.append(row)
        with contextlib.redirect_stdout(io.StringIO()):
            r = ev.evaluate(pl, sems, inss, exp_path=str(out))
        print(f"  {name:28s} {m/len(scenes):9.0f} {100*h/max(t,1):9.1f}% "
              f"{r['all_ap']*100:8.3f} {r['all_ap_50%']*100:8.3f} {r['all_ap_25%']*100:8.3f} "
              f"{r['all_rc_25%']*100:8.3f} {npr:7,d}", flush=True)
    print(f"\n  {t} eligible GT instances over {len(scenes)} scene(s)")


if __name__ == "__main__":
    main()
