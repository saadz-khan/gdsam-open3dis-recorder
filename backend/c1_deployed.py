#!/usr/bin/env python
"""The deployed Component 1 pipeline, in one entry point.

WHY THIS FILE EXISTS.  The headline configuration is not `v96` alone.  Two mechanisms that the
ablation credits are installed outside the config table, and a run that imports `c1_apdecomp`
directly silently omits both:

  * the coarse-segment linking prior (`c1_coarse_linking`, alpha_c = 0.30, -1.06 AP if absent),
    which monkey-patches `link_masks` and must be installed BEFORE anything binds it;
  * mass retention in the partition (`c1_absorb`, retention="mass"), without which the pipeline
    reproduces the count-retention control 41.52 / 62.87 / 76.00 rather than the headline
    41.65 / 63.13 / 76.21.

Import order is load-bearing.  `c1_coarse_linking.install()` also wraps `c1_apdecomp._predict_scene`
with the tracer that records which scene's coarse labels to use, so it has to run before any caller
binds that function.  Installing it lazily from inside `_predict_scene`, or at
`c1_bridge_ablation` import time, does NOT work -- the circular import leaves the patch overwritten,
and the failure is silent: predictions differ by a few proposals per scene with no error raised.
Verified: install-first reproduces the frozen store exactly on 6/6 scenes; both lazy variants give
1/6.

  PYTHONPATH=/home/saad/Desktop/spacesculptor_old <sp311>/python c1_deployed.py --scenes 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("C1_REPO", str(BASE.parent)))
SHARED = os.environ.get("C1_BASE", "/nonexistent-shared-checkout")


def _own_the_path() -> None:
    """Evict the shared baselines checkout and re-assert this worktree, permanently.

    `official_eval` hard-codes the shared baselines directory and prepends it to sys.path at import
    time, so ANY module imported afterwards -- `c1_bridge_ablation` included -- silently loads from
    that checkout instead of from here.  The shared copy has no `coarse_weight` key, so the coarse
    prior was skipped and the run reported `coarse prior 0.00` with no error raised.  Import order
    alone cannot fix this: the poisoning happens on every such import, so the entries have to be
    removed and this worktree re-inserted.
    """
    sys.path[:] = [q for q in sys.path if not str(q).startswith(SHARED)]
    for q in (REPO, BASE, BASE / "Open3DIS"):
        r = str(q)
        while r in sys.path:
            sys.path.remove(r)
        sys.path.insert(0, r)


_own_the_path()
if not hasattr(np, "in1d"):
    np.in1d = np.isin

COARSE_WEIGHT = 0.30          # c1_coarse_full312.json: best_weight
THETA = 0.65                  # frozen V55 manifest
RETENTION = "mass"            # vertex-weighted, matching the official IoU


def install(frontend: str = "v96"):
    """Install every out-of-config mechanism the deployed configuration needs, in order.

    Every step is asserted.  A silent no-op here costs about 1 AP and raises nothing, which is
    exactly how the first version of this file reported `coarse prior 0.00` on a full run.
    """
    _own_the_path()
    import c1_bridge_ablation as cba
    if not str(Path(cba.__file__).resolve()).startswith(str(BASE)):
        raise RuntimeError(f"c1_bridge_ablation shadowed by {cba.__file__}; this worktree must win")
    w = float(cba.FRONTENDS[frontend].get("coarse_weight", 0.0) or 0.0)
    if frontend == "v96" and w <= 0:
        raise RuntimeError("v96 must declare coarse_weight; the headline depends on it")
    if w > 0:
        import c1_coarse_linking as cl
        cl.install(w)
        if getattr(cba.link_masks, "__name__", "") != "patched":
            raise RuntimeError("coarse prior failed to install")
    return w


def predict(scenes, frontend: str = "v96", method: str = "relift", discovery: bool = False,
            joint: bool = False):
    """Frozen predictions for `scenes`, with the prior installed and the partition applied."""
    w = install(frontend)
    import c1_apdecomp as ap
    import official_eval as oe
    from c1_absorb import build as absorb

    store, spps = {}, {}
    for i, s in enumerate(scenes, 1):
        preds, nV = ap._predict_scene(s, frontend, method)
        if preds is None:
            continue
        store[s] = {"preds": preds, "nV": nV}
        spps[s] = oe._superpoints(s, nV)
        if i % 25 == 0:
            print(f"    {i}/{len(scenes)}", flush=True)
    used = sorted(store)
    result = absorb(store, used, spps, THETA, retention=RETENTION)
    if discovery or joint:
        from c1_bridge_ablation import FRONTENDS
        from c1_candidate import _inference_item
        from c1_residual_discovery import discover
        cfg = dict(FRONTENDS[frontend])
        for scene in used:
            item = _inference_item(scene, result[scene]["nV"], cfg,
                                   Path(cfg["recordings"]), True)
            if item is None:
                raise RuntimeError(f"Missing inference input for {scene}")
            result[scene], _ = discover(item, result[scene])
            if joint:
                from c1_information_selected import refine_scene
                result[scene], _ = refine_scene(item, result[scene])
    return result, used, w


def main() -> None:
    pr = argparse.ArgumentParser()
    pr.add_argument("--frontend", default="v96")
    pr.add_argument("--method", default="relift")
    pr.add_argument("--discovery", action="store_true", help="enable validated residual object discovery")
    pr.add_argument("--joint", action="store_true", help="enable validated joint scene information inference, including discovery")
    pr.add_argument("--scenes", type=int, default=0, help="0 = the full 312 split")
    pr.add_argument("--benchmarks", nargs="+", default=["scannetv2", "scannet200"])
    args = pr.parse_args()

    import c1_apdecomp as ap
    split = ap.read_split("full312")
    scenes = split[:args.scenes] if args.scenes else split
    print(f"  {args.frontend} / {args.method} on {len(scenes)} scenes", flush=True)

    store, used, w = predict(scenes, args.frontend, args.method, discovery=args.discovery, joint=args.joint)
    print(f"  coarse prior {w:.2f}, theta {THETA}, retention {RETENTION}", flush=True)
    print(f"  {sum(len(v['preds']) for v in store.values())} proposals over {len(used)} scenes\n",
          flush=True)

    for bench in args.benchmarks:
        gts = ap.ground_truth(used, store, bench)          # GT only AFTER predictions are frozen
        ok = [s for s in used if s in gts]
        r = ap.fast_score(ap.assignments(store, gts, ok, bench), gts, ok, bench)
        print(f"  {bench:11s} AP {r['ap']:6.2f}   AP50 {r['ap50']:6.2f}   AP25 {r['ap25']:6.2f}",
              flush=True)


if __name__ == "__main__":
    main()
