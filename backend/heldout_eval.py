#!/usr/bin/env python
"""HELD-OUT generalisation test with a FROZEN configuration.

The v6 hyperparameters (tau_same .50 / tau_cross .80 / carve .20 / soft 2 / greedy / rank full) were
selected on the 8 tuning scenes. This script applies them UNCHANGED to 16 disjoint ScanNet v2 *val*
scenes and scores ours vs MaskClustering under the identical protocol (same SAM3 masks, same frames
stride 10 / <=200, AABB boxes, class-agnostic, GT-centric best match + ranked all-point AP).

Nothing here is tuned. If the standing holds, the method generalises; if not, we report that.

  PYTHONPATH=<repo> <sp311>/python heldout_eval.py [--clouds <dir>] [--per-scene]
"""
import argparse, glob, os, pickle, sys
import numpy as np
sys.path.insert(0, "/home/saad/Desktop/spacesculptor_old")
sys.path.insert(0, "/home/saad/Desktop/spacesculptor_baselines")
from scripts.scannet_io import load_gt_instances, _read_ply_xyz, load_axis_align
from v6_cluster import make_preds, metrics, iou           # frozen implementation, imported as-is

SCAN = "/home/saad/Desktop/spacesculptor_old/datasets/scannet_raw/scans"
MCROOT = "/home/saad/Desktop/spacesculptor_baselines/MaskClustering/data/scannet/processed"
HELD = "/home/saad/Desktop/spacesculptor_old/runs/scannet_ap/v5_heldout"

# ---- FROZEN CONFIG (selected on the 8 tuning scenes; not re-tuned here) ----
CFG = dict(tau_same=0.50, tau_cross=0.80, veto=0.05, use_veto=True,
           min_masks=1, nms=0.5, rank="full", carve=0.20, contain_nms=1.0,
           soft=2.0, support_w=1.0, greedy=True)


def mc_preds(scene, V):
    f = f"{MCROOT}/{scene}/output/object/scannet/object_dict.npy"
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True).item()
    out = []
    for v in d.values():
        pid = np.asarray(list(v["point_ids"]), int)
        if len(pid) < 50:
            continue
        P = V[pid]
        if float((P.max(0) - P.min(0)).max()) > 4.0:
            continue
        lo, hi = P.min(0), P.max(0)
        out.append((lo, hi, float(len(pid))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", default=HELD)
    ap.add_argument("--per-scene", action="store_true")
    args = ap.parse_args()
    pkls = sorted(glob.glob(f"{args.clouds}/*.pkl"))
    if not pkls:
        print(f"no recorded scenes in {args.clouds}"); return
    ours_all, mc_all, gts_all, names = [], [], [], []
    for f in pkls:
        sc = os.path.basename(f)[:-4]
        sd = f"{SCAN}/{sc}"
        inst, _ = load_gt_instances(sd, sc)
        if not inst:
            continue
        gl = [(np.asarray(g["center"], float), np.asarray(g["size"], float)) for g in inst]
        V0 = _read_ply_xyz(f"{sd}/{sc}_vh_clean_2.ply")
        V = (np.c_[V0, np.ones(len(V0))] @ load_axis_align(f"{sd}/{sc}.txt").T)[:, :3]
        d = pickle.load(open(f, "rb"))
        op = make_preds(d, CFG["tau_same"], CFG["tau_cross"], CFG["veto"], CFG["use_veto"],
                        min_masks=CFG["min_masks"], nms=CFG["nms"], rank=CFG["rank"],
                        carve=CFG["carve"], contain_nms=CFG["contain_nms"],
                        soft=CFG["soft"], support_w=CFG["support_w"], greedy=CFG["greedy"])
        mp = mc_preds(sc, V)
        ours_all.append(op); gts_all.append(gl); names.append(sc)
        mc_all.append(mp if mp is not None else [])
    n_gt = sum(len(g) for g in gts_all)
    have_mc = sum(1 for m in mc_all if m)
    print(f"  === HELD-OUT ({len(names)} disjoint ScanNet val scenes, {n_gt} GT) -- FROZEN config ===")
    print(f"  config: tau {CFG['tau_same']}/{CFG['tau_cross']}  carve {CFG['carve']}  soft {CFG['soft']}"
          f"  greedy {CFG['greedy']}  rank {CFG['rank']}   [selected on the 8 tuning scenes, NOT re-tuned]")
    print(f"  {'method':22s} {'#pred':>6s} {'mIoU':>6s} {'R@.25':>6s} {'R@.5':>6s} {'AP@25':>6s} {'AP@50':>6s} {'interp':>7s}")
    if have_mc:
        m = metrics(mc_all, gts_all)
        print(f"  {'MaskClustering':22s} {m['npred']:6d} {m['miou']:6.3f} {m['r25']:6.3f} {m['r50']:6.3f} "
              f"{m['ap25']:6.1f} {m['ap50']:6.1f} {m['interp']:6.1f}%")
    else:
        print(f"  {'MaskClustering':22s}  (not yet run on held-out scenes)")
    o = metrics(ours_all, gts_all)
    print(f"  {'OURS v6 (frozen)':22s} {o['npred']:6d} {o['miou']:6.3f} {o['r25']:6.3f} {o['r50']:6.3f} "
          f"{o['ap25']:6.1f} {o['ap50']:6.1f} {o['interp']:6.1f}%")
    if args.per_scene:
        print(f"\n  {'scene':16s} {'GT':>4s} | {'MC mIoU':>8s} {'ours':>7s} | {'MC AP25':>8s} {'ours':>7s}")
        for i, sc in enumerate(names):
            om = metrics([ours_all[i]], [gts_all[i]])
            mm = metrics([mc_all[i]], [gts_all[i]]) if mc_all[i] else None
            a = f"{mm['miou']:8.3f}" if mm else "     n/a"
            b = f"{mm['ap25']:8.1f}" if mm else "     n/a"
            print(f"  {sc:16s} {len(gts_all[i]):4d} | {a} {om['miou']:7.3f} | {b} {om['ap25']:7.1f}")


if __name__ == "__main__":
    main()
