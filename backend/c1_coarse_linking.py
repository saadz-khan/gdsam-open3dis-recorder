#!/usr/bin/env python
"""COARSE-SEGMENT AGREEMENT AS ADDITIVE EVIDENCE INSIDE MASK LINKING.

WHY.  A matched-front-end ablation (C1_RESEARCH_LOG 2.11) put our fusion ~4-5 AP BEHIND MV3DIS once
both systems consume identical Grounded-SAM masks: our near-parity on ScanNetV2 is largely the front
end, not the linker.  The stated mechanistic difference is the UNIT of decision.  MV3DIS agglomerates
a few COARSE 3D segments -- broad, robust, geometry-derived -- whereas ``link_masks`` agglomerates
MANY FINE 2D masks joined by pairwise 3D containment, so every individual decision is noisier even
though the population of decisions is far denser.

THE CHANGE.  Keep the fine containment mechanism as the primary and only merge driver, and add the
coarse geometry as a SECOND, PURELY ADDITIVE piece of evidence in the merge-candidacy test.  For a
pair of clusters let ``coarse_agree`` be the fraction of their vertices lying in shared coarse
segments (the exact analogue of ``cont``, computed against a coarse mesh partition instead of against
pairwise mask overlap).  Then

    thr_effective = max(thr - coarse_weight * coarse_agree, 0)
    cand          = cont >= thr_effective

Two clusters that are different surface patches of the SAME coarse geometric part need less direct
containment evidence to merge; two clusters straddling a real geometric discontinuity get no help at
all.  At ``coarse_weight = 0`` the expression collapses to ``cont >= thr`` and the method is the
baseline exactly.

WHAT THIS DELIBERATELY IS NOT.  Two neighbouring ideas are already refuted and are not repeated here:
(a) coarse-guide mask FILTERING -- no mask is dropped, filtered or reweighted by this mechanism;
(b) a standalone superpoint graph -- the coarse partition never becomes the unit of agglomeration,
    it only supplies a bounded discount on an existing threshold.
The view-consensus veto is untouched.  It is applied AFTER candidacy as ``cand &= ~veto``, so a pair
the veto blocks stays blocked no matter how strong its coarse agreement is; the coarse term can only
ever relax the POSITIVE threshold.

  PYTHONPATH=<repo> <env>/python c1_coarse_linking.py --tune-weights 0.0 0.05 0.1 0.2 0.3
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

BASE = Path(__file__).resolve().parent
REPO = Path("/home/saad/Desktop/spacesculptor_old")
for _p in (REPO, BASE, BASE / "Open3DIS"):
    sys.path.insert(0, str(_p))
if not hasattr(np, "in1d"):
    np.in1d = np.isin

SHARED = "/home/saad/Desktop/spacesculptor_baselines"


def _own_the_path():
    """Keep THIS worktree ahead of the shared baselines checkout, permanently.

    ``official_eval`` and ``mesh_oversegment`` both HARD-CODE the shared baselines directory and
    prepend it to ``sys.path`` at import time.  Anything imported afterwards -- including the
    lazily-imported ``c1_candidate`` / ``c1_relift_refinement`` that carry the actual inference --
    then silently loads from the shared checkout rather than from this worktree.  Import order
    alone cannot fix that, because the poisoning happens on every such import; the shared entries
    have to be evicted and this worktree re-asserted.
    """
    sys.path[:] = [p for p in sys.path if not str(p).startswith(SHARED)]
    for p in (REPO, BASE, BASE / "Open3DIS"):
        s = str(p)
        while s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


import c1_apdecomp as ap  # noqa: E402
_own_the_path()
import official_eval as oe  # noqa: E402
_own_the_path()
import mesh_oversegment as mo  # noqa: E402
_own_the_path()
import c1_bridge_ablation as cba  # noqa: E402
from v6_cluster import _consensus_ratio  # noqa: E402
_own_the_path()

# Chosen in c1_coarse_calibrate.py by a size criterion only; AP was never consulted.  See report.
COARSE_K = 0.05
COARSE_MIN_VERTS = 100
COARSE_DIR = BASE / "coarse_cache"

_ORIGINAL_LINK_MASKS = cba.link_masks
_ORIGINAL_PREDICT_SCENE = ap._predict_scene
_CURRENT = {"scene": None}
_LABELS: dict[str, np.ndarray] = {}
# A dense reference implementation is O(nC^2 * n_seg) and only tractable on small scenes, so the
# gate is cross-checked against it on the two lowest-mask-count TUNE scenes.
ap.SPLITS["coarse_verify"] = BASE / "splits/coarse_verify_scenes.txt"


# ------------------------------------------------------------------ coarse partition
def coarse_labels(scene: str, k: float = COARSE_K, min_verts: int = COARSE_MIN_VERTS) -> np.ndarray:
    """Felzenszwalb-Huttenlocher partition of the scene mesh at OBJECT-PART scale, cached to disk."""
    out = COARSE_DIR / f"k{k}_m{min_verts}" / f"{scene}.npy"
    if out.exists():
        return np.load(out)
    V, F = mo.read_ply_mesh(f"{oe.SCANS}/{scene}/{scene}_vh_clean_2.ply")
    lab = mo.oversegment(V, F, k=k, min_verts=min_verts).astype(np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, lab)
    return lab


def _pair_agreement(Hd, sz, ai, bi, chunk=20_000):
    """sum_s min(H[a,s], H[b,s]) / min(sz_a, sz_b) for the listed pairs -- the analogue of ``cont``.

    ``cont`` is |A n B| / min(|A|,|B|): direct vertex overlap.  This is the same ratio with the
    intersection taken in COARSE-SEGMENT space, so two clusters covering disjoint patches of one
    coarse part still score high, which is precisely the evidence the fine containment cannot see.
    """
    out = np.empty(len(ai), np.float64)
    for s in range(0, len(ai), chunk):
        a, b = ai[s:s + chunk], bi[s:s + chunk]
        out[s:s + chunk] = np.minimum(Hd[a], Hd[b]).sum(1)
    return out / np.minimum(sz[ai], sz[bi])


def link_masks_coarse(d, tau_same=0.45, tau_cross=0.75, veto=0.05, max_ext=3.0, use_veto=True,
                      carve=0.0, greedy=False, return_members=False, claim_mode="frame",
                      veto_mode="mean", conflict_frac=0.5, return_hierarchy=False,
                      coarse_weight=0.0, coarse=None, dense_agree=False,
                      spp=None, kappa=0.0, split_ncut=0.0, split_min_side=2,
                      split_mode="replace"):
    """``v6_cluster.link_masks`` with a bounded coarse-agreement discount on the merge threshold.

    Every line is the original except the three marked ``COARSE`` below.  The veto block is verbatim.
    """
    # v6_cluster.link_masks grew five parameters after this variant was written.  All are inert at
    # their DEPLOYED values (segment shrinkage off, n-cut splitting off), so accepting and asserting
    # them is exact -- but silently swallowing a non-default would run a different algorithm than the
    # caller asked for, so refuse instead.
    if spp is not None and kappa > 0:
        raise NotImplementedError("segment shrinkage (kappa > 0) is not implemented in the "
                                  "coarse-discounted linker; it is off in the deployed config")
    if split_ncut:
        raise NotImplementedError("n-cut splitting is not implemented in the coarse-discounted "
                                  "linker; it is off in the deployed config")

    P = d["P"].tocsc(); nM = d["nM"]
    if nM == 0: return []
    V = d["V"]; conc = d["mask_concept"].astype(np.int32); mf = d["mask_frame"]
    Vis = d["Vis"]
    vc = (P.T @ Vis).toarray(); size0 = np.asarray(P.sum(0)).ravel() + 1e-9
    vf = (vc / size0[:, None]) > 0.25
    ov0 = (P.T @ P).toarray()
    cont0 = ov0 / (np.minimum.outer(size0, size0) + 1e-9)
    obs = vf.astype(np.float32) @ vf.T.astype(np.float32)
    cm = (ov0 / (vc[:, mf] + 1e-9)) > 0.8
    sup = cm.astype(np.float32) @ cm.astype(np.float32).T
    rate0 = sup / (obs + 1e-7)
    if veto_mode == "conflict-fraction":
        eligible0 = (obs >= 3).astype(np.float32)
        conflict0 = (eligible0 * (rate0 < veto)).astype(np.float32)

    # COARSE (1/3): per-vertex coarse-segment indicator over P's row space, built once per scene.
    S = None
    if coarse_weight > 0.0 and coarse is not None:
        lab = np.full(P.shape[0], -1, np.int64)
        n = min(len(coarse), P.shape[0])
        lab[:n] = coarse[:n]
        ok = lab >= 0
        n_seg = int(lab[ok].max()) + 1 if ok.any() else 0
        if n_seg:
            S = sp.csr_matrix((np.ones(int(ok.sum()), np.float32),
                               (np.flatnonzero(ok), lab[ok])), shape=(P.shape[0], n_seg))

    G = np.arange(nM)
    hierarchy = {tuple((mask,)) for mask in range(nM)} if return_hierarchy else None
    for _ in range(30):
        cl_ids = np.unique(G); nC = len(cl_ids)
        if nC <= 1: break
        remap = {c: i for i, c in enumerate(cl_ids)}
        idx = np.array([remap[g] for g in G])
        Gm = sp.csr_matrix((np.ones(nM, np.float32), (np.arange(nM), idx)), shape=(nM, nC))
        Pc = (P @ Gm).tocsc(); Pc.data[:] = 1.0
        sz = np.asarray(Pc.sum(0)).ravel() + 1e-9
        ov = (Pc.T @ Pc).toarray()
        cont = ov / (np.minimum.outer(sz, sz) + 1e-9)
        np.fill_diagonal(cont, 0.0)
        cconc = np.zeros(nC, np.int32)
        for c in range(nC):
            mem = conc[idx == c]
            cconc[c] = np.bincount(mem).argmax() if len(mem) else -1
        same = cconc[:, None] == cconc[None, :]
        thr = np.where(same, tau_same, tau_cross)

        # COARSE (2/3): relax the POSITIVE threshold by the pair's coarse agreement.  A pair only
        # needs evaluating when it sits in the band the discount can possibly reach, because
        # coarse_agree in [0,1] gives thr_eff in [max(thr-w,0), thr]: pairs above thr are candidates
        # regardless and pairs below max(thr-w,0) are rejected regardless.  The gate is therefore
        # EXACT, not an approximation, and it makes coarse_weight=0 the untouched baseline by
        # construction (the band is empty).
        cand = cont >= thr
        if S is not None:
            H = (Pc.T @ S).toarray()
            lo = np.maximum(thr - coarse_weight, 0.0)
            if dense_agree:                       # reference path: full matrix, no gating
                agree = np.zeros((nC, nC))
                ii, jj = np.triu_indices(nC, 1)
                agree[ii, jj] = _pair_agreement(H, sz, ii, jj)
                agree += agree.T
                cand = cont >= np.maximum(thr - coarse_weight * agree, 0.0)
            else:
                band = np.triu((cont >= lo) & (cont < thr), 1)
                ii, jj = np.where(band)
                if len(ii):
                    agree = _pair_agreement(H, sz, ii, jj)
                    hit = cont[ii, jj] >= np.maximum(thr[ii, jj] - coarse_weight * agree, 0.0)
                    cand[ii[hit], jj[hit]] = True
                    cand[jj[hit], ii[hit]] = True
            np.fill_diagonal(cand, False)         # no-op at w=0 (cont diagonal is 0 < tau)

        # ---- view-consensus veto: VERBATIM from v6_cluster.link_masks, applied AFTER candidacy ----
        if use_veto:
            Gd = Gm.toarray()
            if veto_mode == "mean":
                ob = Gd.T @ obs @ Gd
                sup_c = Gd.T @ (rate0 * obs) @ Gd
                rate = sup_c / (ob + 1e-7)
                cand &= ~((ob >= 3) & (rate < veto))
            elif veto_mode == "conflict-fraction":
                eligible_c = Gd.T @ eligible0 @ Gd
                conflict_c = Gd.T @ conflict0 @ Gd
                fraction = conflict_c / (eligible_c + 1e-7)
                cand &= ~((eligible_c >= 1) & (fraction >= conflict_frac))
            else:
                raise ValueError(f"unknown veto mode: {veto_mode}")
        # ------------------------------------------------------------------------------------------
        if not cand.any(): break
        score = np.where(cand, cont, -1.0)
        best = score.argmax(1); merged = False
        newG = G.copy()
        done = np.zeros(nC, bool)
        for a in range(nC):
            b = best[a]
            if score[a, b] < 0 or done[a] or done[b] or a == b: continue
            if best[b] != a: continue
            gg = V[np.unique(np.concatenate([Pc[:, a].indices, Pc[:, b].indices]))]
            if len(gg) and (gg.max(0) - gg.min(0)).max() > max_ext: continue
            newG[idx == b] = cl_ids[a]
            done[a] = done[b] = True; merged = True
        if not merged and greedy:
            ii, jj = np.where(np.triu(cand, 1))
            order = np.argsort(-cont[ii, jj])
            for k in order:
                a, b = int(ii[k]), int(jj[k])
                if done[a] or done[b]: continue
                gg = V[np.unique(np.concatenate([Pc[:, a].indices, Pc[:, b].indices]))]
                if len(gg) and (gg.max(0) - gg.min(0)).max() > max_ext: continue
                newG[idx == b] = cl_ids[a]
                done[a] = done[b] = True; merged = True
        if not merged: break
        G = newG
        if return_hierarchy:
            for cluster in np.unique(G):
                hierarchy.add(tuple(np.where(G == cluster)[0].tolist()))
    out = []
    Vis_c = d["Vis"].tocsc()
    if return_hierarchy:
        member_sets = [np.asarray(item, np.int32) for item in sorted(hierarchy, key=lambda x: (len(x), x))]
    else:
        member_sets = [np.where(G == cluster)[0] for cluster in np.unique(G)]
    for members in member_sets:
        vids = np.unique(P[:, members].indices)
        nviews = len(np.unique(mf[members]))
        if carve > 0 and len(vids) >= 40:
            ratio = _consensus_ratio(P, Vis_c, vids, members, mf, claim_mode)
            keep = ratio >= carve
            mcons = float(ratio[keep].mean()) if keep.sum() else 0.0
            if keep.sum() >= 40:
                vids = vids[keep]
        else:
            mcons = 0.5
        out.append((vids, len(members), nviews, mcons) + ((members,) if return_members else ()))
    return out


# ------------------------------------------------------------------ patched inference
def _traced_predict_scene(scene, frontend, method):
    """``link_masks`` never receives the scene name; the per-scene entry point does."""
    _CURRENT["scene"] = scene
    if scene not in _LABELS:
        _LABELS[scene] = coarse_labels(scene)
    return _ORIGINAL_PREDICT_SCENE(scene, frontend, method)


def evaluate(split, frontend, method, annotations, coarse_weight, dense_agree=False):
    """Standard prediction + official scoring path with ``link_masks`` swapped underneath."""
    def patched(d, *a, **kw):
        return link_masks_coarse(d, *a, coarse_weight=coarse_weight,
                                 coarse=_LABELS.get(_CURRENT["scene"]),
                                 dense_agree=dense_agree, **kw)

    _own_the_path()
    cba.link_masks = patched
    ap._predict_scene = _traced_predict_scene
    try:
        ap.CACHE.mkdir(exist_ok=True)
        store = ap.predictions(split, frontend, method, refresh=True)
        scenes = ap.read_split(split)
        out = {}
        for annotation in annotations:
            gts = ap.ground_truth(scenes, store, annotation)
            usable = [s for s in scenes if s in store and s in gts]
            out[annotation] = ap.score(store, gts, usable, annotation)
        out["_store"] = store
        return out
    finally:
        cba.link_masks = _ORIGINAL_LINK_MASKS
        ap._predict_scene = _ORIGINAL_PREDICT_SCENE


def _digest(store):
    """Order-insensitive fingerprint of a prediction set, for bit-for-bit equality checks."""
    import hashlib
    h = hashlib.sha1()
    for scene in sorted(store):
        h.update(scene.encode())
        h.update(str(store[scene]["nV"]).encode())
        for v in sorted(store[scene]["preds"], key=lambda p: (len(p["v"]), int(p["v"][0]) if len(p["v"]) else -1)):
            h.update(v["v"].tobytes())
    return h.hexdigest()[:16]


def _baseline_digest(split, frontend, method):
    """Digest of the store produced by the COMPLETELY UNMODIFIED pipeline (c1_baseline_check.py)."""
    p = BASE / "baseline_store" / f"{split}_{frontend}_{method}.pkl"
    if not p.exists():
        return None
    with p.open("rb") as h:
        return _digest(pickle.load(h))


def _row(name, s):
    return (f"  {name:22s} AP {s['ap']:6.2f}  AP50 {s['ap50']:6.2f}  AP25 {s['ap25']:6.2f}  "
            f"AR {s['ar']:6.2f}  n_pred {s['n_pred']:5d}")


def main():
    pr = argparse.ArgumentParser()
    pr.add_argument("--tune-split", default="tune")
    pr.add_argument("--dev-split", default="dev")
    pr.add_argument("--frontend", default="v96")
    pr.add_argument("--method", default="relift")
    pr.add_argument("--tune-weights", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3])
    pr.add_argument("--annotations", nargs="+", default=["scannetv2", "scannet200"])
    pr.add_argument("--verify-gating", action="store_true",
                    help="cross-check the exact gate against the dense reference")
    pr.add_argument("--verify-split", default="coarse_verify")
    pr.add_argument("--dev-only-weight", type=float, default=None)
    pr.add_argument("--skip-baseline-arm", action="store_true",
                    help="the w=0 arm for this split is already measured and stored")
    pr.add_argument("--json-out", default="c1_coarse_linking.json")
    args = pr.parse_args()

    assert str(ap.__file__).startswith(str(BASE)), "ISOLATION FAILURE"
    print(f"  module {ap.__file__}\n  CACHE  {ap.CACHE}")
    print(f"  coarse partition: FH k={COARSE_K} min_verts={COARSE_MIN_VERTS}\n", flush=True)

    results, digests = {}, {}
    if args.dev_only_weight is None:
        print(f"  === TUNE selection ({args.tune_split}) ===")
        for w in args.tune_weights:
            t0 = time.time()
            row = evaluate(args.tune_split, args.frontend, args.method, args.annotations, w)
            digests[w] = _digest(row.pop("_store"))
            results[w] = row
            cells = "  ".join(f"{a[:9]} AP {row[a]['ap']:6.2f} AP50 {row[a]['ap50']:6.2f} "
                              f"AR {row[a]['ar']:6.2f}" for a in args.annotations)
            print(f"  w={w:5.2f}  {cells}  n={row[args.annotations[0]]['n_pred']:4d} "
                  f"[{digests[w]}] {time.time()-t0:5.0f}s", flush=True)
            if w == 0.0:
                ref = _baseline_digest(args.tune_split, args.frontend, args.method)
                print(f"           w=0 vs UNMODIFIED pipeline: {digests[w]} vs {ref} -> "
                      f"{'BIT-IDENTICAL' if ref == digests[w] else 'DIFFERS (STOP)'}", flush=True)

        if args.verify_gating and len(args.tune_weights) > 1:
            w = max(w for w in args.tune_weights if w > 0)
            print(f"\n  gating exactness check on {args.verify_split} at w={w}", flush=True)
            a = evaluate(args.verify_split, args.frontend, args.method,
                         args.annotations[:1], w, dense_agree=False)
            b = evaluate(args.verify_split, args.frontend, args.method,
                         args.annotations[:1], w, dense_agree=True)
            da, db = _digest(a.pop("_store")), _digest(b.pop("_store"))
            print(f"    gated {da}  dense {db}  "
                  f"{'IDENTICAL' if da == db else 'MISMATCH'}", flush=True)

        primary = args.annotations[0]
        best = max(results, key=lambda w: results[w][primary]["ap"])
        base = results.get(0.0, {}).get(primary, {}).get("ap")
        print(f"\n  best TUNE weight {best:.2f}"
              + (f"  ({results[best][primary]['ap'] - base:+.2f} AP vs w=0)" if base else ""))
        if best == 0.0:
            print("  weight tunes to 0: the mechanism is REFUTED on the tuning split.")
    else:
        best = args.dev_only_weight
        print(f"  (skipping TUNE; validating frozen weight {best})")

    print(f"\n  === DEV validation ({args.dev_split}), weight frozen at {best:.2f} ===", flush=True)
    dev = {}
    for w in ([] if args.skip_baseline_arm else [0.0]) + ([best] if best != 0.0 else []):
        dev[w] = evaluate(args.dev_split, args.frontend, args.method, args.annotations, w)
        d = _digest(dev[w].pop("_store"))
        if w == 0.0:
            ref = _baseline_digest(args.dev_split, args.frontend, args.method)
            print(f"  w=0 vs UNMODIFIED pipeline: {d} vs {ref} -> "
                  f"{'BIT-IDENTICAL' if ref == d else 'DIFFERS (STOP)'}", flush=True)
        for a in args.annotations:
            print(_row(f"w={w:.2f} {a}", dev[w][a]), flush=True)
    if best != 0.0 and 0.0 in dev:
        print(f"\n  {'annotation':12s} {'metric':>7s} {'w=0.00':>9s} {'w=%.2f' % best:>9s} {'delta':>8s}")
        for a in args.annotations:
            for m in ("ap", "ap50", "ap25", "ar"):
                x, y = dev[0.0][a][m], dev[best][a][m]
                print(f"  {a:12s} {m:>7s} {x:9.2f} {y:9.2f} {y - x:+8.2f}")

    Path(args.json_out).write_text(json.dumps(
        {"tune": {str(k): v for k, v in results.items()}, "digests": {str(k): v for k, v in digests.items()},
         "best_weight": best, "dev": {str(k): v for k, v in dev.items()},
         "coarse": {"k": COARSE_K, "min_verts": COARSE_MIN_VERTS}}, indent=2, sort_keys=True) + "\n")
    print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# Deployed entry point.  The headline configuration runs WITH this prior (alpha_c = 0.30), and it
# used to be installed by importing this module from a separate worktree -- which meant the branch
# shipped as the reproducible system silently ran without it.  `install()` makes the dependency
# explicit: c1_bridge_ablation calls it when a frontend config carries "coarse_weight".
# ---------------------------------------------------------------------------------------------
DEPLOYED_WEIGHT = 0.30            # selected on the full split; c1_coarse_full312.json best_weight


def install(coarse_weight: float = DEPLOYED_WEIGHT, dense_agree: bool = False) -> None:
    """Swap the coarse-discounted linker in underneath the standard prediction path."""
    def patched(d, *a, **kw):
        return link_masks_coarse(d, *a, coarse_weight=coarse_weight,
                                 coarse=_LABELS.get(_CURRENT["scene"]),
                                 dense_agree=dense_agree, **kw)
    _own_the_path()
    cba.link_masks = patched
    ap._predict_scene = _traced_predict_scene


def uninstall() -> None:
    """Restore the unpatched linker; coarse_weight = 0 is the ablation control."""
    cba.link_masks = _ORIGINAL_LINK_MASKS
    ap._predict_scene = _ORIGINAL_PREDICT_SCENE
