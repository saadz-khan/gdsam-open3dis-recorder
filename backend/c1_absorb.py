#!/usr/bin/env python
"""Absorb the slivers the partition creates, instead of emitting them.

THE FINDING THIS ACTS ON.  Characterising the dominated set D -- after seventeen feature searches
failed to detect it -- showed what it actually is:

  93.6%  lie ENTIRELY INSIDE one annotated instance, touching no other
  median size  3.8% of that instance
  62.7%  sit on an instance we DID recover correctly with another proposal
   0%    overlap any matched proposal at all

The last two are only consistent because of the partition itself.  It awards every superpoint to its
highest-confidence claimant, so the emitted proposals are disjoint by construction.  On one object
the main proposal wins most superpoints and a competing fragment proposal wins a handful -- and those
handful become a separate, disjoint, charged false positive.  The step that removes duplicates is
manufacturing slivers.

THE FIX IS STRUCTURAL, NOT A QUALITY SIGNAL, WHICH IS WHY THE FEATURE SEARCHES COULD NOT FIND IT.
A proposal that entered the partition claiming a large region and leaves it holding a sliver has not
been confirmed -- it has been outvoted almost everywhere.  Rather than emitting the remnant, release
its superpoints to the next-best claimant, which is the proposal that already owns the rest of that
object.  A true object is not affected: it wins most of what it claimed.

    retained(i) = mass of superpoints i won / mass of superpoints i claimed
    if retained(i) < theta:  release i's superpoints to their runner-up

RETENTION.  "mass" (default, DEPLOYED) weights each superpoint by its vertex count, because the
official ScanNet IoU is vertex-weighted and superpoints differ substantially in size: equal votes
can otherwise drop a proposal that kept most of its surface, or keep one that lost most of it.
"count" is the unweighted rule and is retained as the exact control -- it reproduces the previously
reported 41.52 / 62.87 / 76.00 on ScanNetV2, against 41.65 / 63.13 / 76.21 for mass.  Nothing else
differs between the two: eligibility, confidence arbitration, the round cap, the emission floor and
theta are shared.

Iterated to a fixed point, since releasing one proposal can change another's retention.  Nothing is
deleted for being low quality; a proposal only loses ground it failed to hold.

  PYTHONPATH=... <env>/python c1_absorb.py --thetas 0.2 0.35 0.5 0.65
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = Path("/home/saad/Desktop/spacesculptor_old")
for _p in (REPO, BASE, BASE / "Open3DIS"):
    sys.path.insert(0, str(_p))
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import c1_apdecomp as ap  # noqa: E402


def partition_absorb(entry, spp, theta, min_share=0.5, min_pts=100, max_iter=4,
                     arbitrate="conf", retention="mass"):
    """The promoted partition, plus: a proposal that fails to hold its ground releases it.

    theta = 0 reproduces the promoted partition exactly, which is how the ablation is controlled.

    ARBITRATE decides who wins a contested superpoint, and the default has a size bias.

      "conf"  (default, unchanged) ranks claimants by the proposal confidence, which is
              views * (1 - exp(-|V| / 200)) * purity**expo.  That factor saturates at 1 for a large
              proposal and sits at 0.63 for a 200-vertex one, so a big proposal outranks a small
              one at every superpoint they contest -- regardless of which of them actually covers
              it.  The auction is decided by who is bigger, not whose superpoint it is.  Measured
              consequence on 25 scenes: missed instances have median size 1526 vertices against
              2784 for reached ones, 47% of misses fall in the smallest quartile of all instances,
              and 30.7% of missed instances have their surface sitting inside a proposal owned by a
              LARGER instance.  A swallowed object emits nothing, which is the UNTOUCHED bin.

      "share" ranks by the fraction of THAT superpoint the proposal covers -- already computed as
              `shares` for the eligibility test, and never used for the award.  Size drops out.

      "share_conf" multiplies the two, keeping confidence only as a tie-break between claimants
              that cover the superpoint about equally.

    Default is "conf", so every existing number is reproduced bit-for-bit.
    """
    preds = entry["preds"]
    if not preds or spp is None:
        return [p["v"] for p in preds]
    n_spp = int(spp.max()) + 1
    tot = np.bincount(spp[spp >= 0], minlength=n_spp).astype(np.float64)

    # claims[i] = superpoints proposal i is eligible for, with its confidence
    shares = np.zeros((len(preds), n_spp), np.float32)
    conf = np.array([p["conf"] for p in preds], np.float64)
    for i, p in enumerate(preds):
        lab = spp[p["v"]]
        lab = lab[lab >= 0]
        if lab.size:
            shares[i] = np.bincount(lab, minlength=n_spp) / np.maximum(tot, 1.0)
    eligible = shares >= min_share
    if retention not in {"count", "mass"}:
        raise ValueError(f"unknown retention rule: {retention}")
    # mass: weight every eligible superpoint by its vertex count.  count: unit weights.
    claim_weight = (eligible * tot[None, :]) if retention == "mass" else eligible.astype(np.float64)
    claimed = claim_weight.sum(1)

    if arbitrate == "conf":
        rank = np.broadcast_to(conf[:, None], shares.shape)
    elif arbitrate == "share":
        rank = shares.astype(np.float64)
    elif arbitrate == "share_conf":
        rank = shares.astype(np.float64) * conf[:, None]
    else:
        raise ValueError(f"unknown arbitrate rule: {arbitrate}")

    alive = np.ones(len(preds), bool)
    for _ in range(max_iter):
        # award each superpoint to the highest-ranked LIVING eligible claimant
        score = np.where(eligible & alive[:, None], rank, -np.inf)
        owner = np.where(np.isfinite(score.max(0)), score.argmax(0), -1)
        won = np.array([claim_weight[i, owner == i].sum() for i in range(len(preds))], np.float64)
        retained = np.divide(won, np.maximum(claimed, 1.0), where=claimed > 0,
                             out=np.zeros(len(preds)))
        drop = alive & (claimed > 0) & (retained < theta)
        if not drop.any():
            break
        alive &= ~drop

    score = np.where(eligible & alive[:, None], rank, -np.inf)
    owner = np.where(np.isfinite(score.max(0)), score.argmax(0), -1)
    out = []
    for i in range(len(preds)):
        own = np.flatnonzero(owner == i)
        if own.size == 0 or not alive[i]:
            out.append(np.empty(0, np.int32))
            continue
        keep = np.flatnonzero(np.isin(spp, own)).astype(np.int32)
        out.append(keep if keep.size >= min_pts else np.empty(0, np.int32))
    return out


def build(store, scenes, spps, theta, arbitrate="conf", retention="mass"):
    new = {}
    for s in scenes:
        vs = partition_absorb(store[s], spps.get(s), theta, arbitrate=arbitrate,
                              retention=retention)
        preds = [{"v": v, "conf": p["conf"]} for v, p in zip(vs, store[s]["preds"]) if v.size]
        new[s] = {"preds": preds, "nV": store[s]["nV"]}
    return new


def main() -> None:
    pr = argparse.ArgumentParser()
    pr.add_argument("--thetas", nargs="+", type=float, default=[0.0, 0.2, 0.35, 0.5, 0.65, 0.8])
    pr.add_argument("--json-out", default="absorb.json")
    args = pr.parse_args()

    import official_eval as oe
    from c1_bridge_ablation import FRONTENDS
    FRONTENDS.setdefault("v96coarse", dict(FRONTENDS["v96"]))

    # start from the PRE-partition store, so theta=0 must reproduce the promoted partition exactly
    store = ap.predictions("full312", "v96coarse", "relift")
    scenes = [s for s in ap.read_split("full312") if s in store]
    spps = {}
    for i, s in enumerate(scenes):
        sp = oe._spp_cache.get(s)
        if sp is None:
            sp = oe._superpoints(s, store[s]["nV"])
            oe._spp_cache[s] = sp
        spps[s] = sp
        if (i + 1) % 100 == 0:
            print(f"    superpoints {i+1}/{len(scenes)}", flush=True)
    print(f"  {len(scenes)} scenes\n", flush=True)

    halves = {"A": scenes[0::2], "B": scenes[1::2]}
    out = {"thetas": {}}
    for annotation in ("scannetv2", "scannet200"):
        gts = ap.ground_truth(scenes, store, annotation)
        sc = [s for s in scenes if s in gts]
        print(f"  === {annotation} ===")
        print(f"  {'theta':>6s} {'n_pred':>7s} {'mAP':>7s} {'AP50':>7s} {'AP25':>7s} | "
              f"{'A mAP':>7s} {'A AP25':>7s} | {'B mAP':>7s} {'B AP25':>7s}")
        for th in args.thetas:
            st = build(store, sc, spps, th)
            asg = ap.assignments(st, gts, sc, annotation)
            r = ap.fast_score(asg, gts, sc, annotation)
            cells = []
            for tag, sub in halves.items():
                sub = [s for s in sub if s in asg]
                h = ap.fast_score(asg, gts, sub, annotation)
                cells.append(f"{h['ap']:7.2f} {h['ap25']:7.2f}")
            print(f"  {th:6.2f} {r['n_pred']:7d} {r['ap']:7.2f} {r['ap50']:7.2f} {r['ap25']:7.2f} | "
                  f"{cells[0]} | {cells[1]}", flush=True)
            out["thetas"].setdefault(annotation, {})[f"{th:.2f}"] = {
                k: r[k] for k in ("ap", "ap50", "ap25", "n_pred")}
        print()

    Path(args.json_out).write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
