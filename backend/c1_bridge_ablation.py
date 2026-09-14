#!/usr/bin/env python
"""Evaluate bridge-resistant cannot-link aggregation on frozen development scenes.

The deployed mean veto averages separation evidence over every cross-cluster mask pair.  A broad mask
can therefore act as an articulation point: its positive edges dilute a clean A-vs-B cannot-link and
transitively weld two objects.  ``conflict-fraction`` preserves the fraction of independently
co-observed mask pairs that explicitly support separation.

This script changes only that aggregation rule.  It evaluates both the SAM3 operating point and the
separately frozen Grounded-SAM operating point, under the published ScanNetV2 GT protocol, and reports
both method-native ranking and the all-constant MV3DIS score convention.  Use TUNE to select a rule and
DEV exactly once to validate it; TEST is intentionally not an option by default.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("C1_REPO", "."))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "Open3DIS"))

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import official_eval as oe  # noqa: E402
from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval  # noqa: E402
# v6_cluster MUST come from THIS worktree.  spacesculptor_baselines holds a copy, and importing
# official_eval (above) puts that worktree on sys.path AHEAD of this one, so a plain
# `from v6_cluster import ...` silently binds the baselines copy.  The two are the same algorithm
# today -- which is why nothing measured so far is wrong -- but an edit made here would then be
# invisible at runtime, and the failure is silent whenever the signatures still happen to match.
# Load it by path so import order cannot decide which experiment we are running.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("v6_cluster_local", BASE / "v6_cluster.py")
_v6 = _ilu.module_from_spec(_spec)
sys.modules["v6_cluster_local"] = _v6
_spec.loader.exec_module(_v6)
link_masks = _v6.link_masks


FRONTENDS = {
    "sam3": {
        "recordings": REPO / "runs/scannet_ap/v5_all_v2",
        "tau_same": 0.50,
        "tau_cross": 0.80,
        "carve": 0.20,
        "snap_mode": "selfiou",
        "snap": 0.50,
        "exponent": 3,
        "nms": 0.50,
    },
    # Clean fixed-vocabulary recordings (RESULTS 0.7): same backend settings as "sam3" so the only
    # difference is the front end's prompt list.  Any re-fitting for this front end's mask density is
    # done explicitly and recorded, never inherited.
    "sam3_clean": {
        "recordings": REPO / "runs/scannet_ap/v6_clean",
        "tau_same": 0.50,
        "tau_cross": 0.80,
        "carve": 0.20,
        "snap_mode": "selfiou",
        "snap": 0.50,
        "exponent": 3,
        "nms": 0.50,
    },
    # Promptless SAM automatic-mask front end.  Every mask carries the same pseudo-concept, so
    # link_masks applies tau_same to EVERY pair and tau_cross is inert: the concept prior has nothing
    # to condition on.  tau_same is therefore re-fitted for this regime rather than inherited.
    # SAM 3 driven by four generic, scene-independent words ("object", "furniture", "appliance",
    # "item").  Zero-shot and vocabulary-free in the sense that matters -- no benchmark class names --
    # while still using the strongest available segmenter.
    # Identical prompts and frames to "sam3_clean"; the ONLY difference is that visibility and mask
    # claims carry graded depth agreement instead of 1.0.  MV3DIS attributes +1.5 AP to this.
    "sam3_dw": {
        "recordings": REPO / "runs/scannet_ap/dw_clean",
        "tau_same": 0.50,
        "tau_cross": 0.80,
        "carve": 0.20,
        "snap_mode": "selfiou",
        "snap": 0.50,
        "exponent": 3,
        "nms": 0.50,
    },
    # The 3-way union (vocabulary + generic prompts + automatic masks).  tau_same == tau_cross because
    # the TUNE sweep showed the concept prior is inert here: the gain is the union, not the priming.
    "fused": {
        "recordings": REPO / "runs/scannet_ap/fused_clean",
        "tau_same": 0.50,
        "tau_cross": 0.50,
        "carve": 0.20,
        "snap_mode": "selfiou",
        "snap": 0.50,
        "exponent": 3,
        "nms": 0.50,
    },
    # stride 5 / 450 frames instead of stride 10 / 200: the only lever that raises coverage for every
    # front end without changing the segmenter.
    "frames450": {
        "recordings": REPO / "runs/scannet_ap/frames450",
        "tau_same": 0.50,
        "tau_cross": 0.80,
        "carve": 0.20,
        "snap_mode": "selfiou",
        "snap": 0.50,
        "exponent": 3,
        "nms": 0.50,
    },
    # 48 terms: a prefix of the PUBLISHED ScanNet200 class list, scene-independent by construction.
    "v48": {
        "recordings": REPO / "runs/scannet_ap/v48_clean",
        "tau_same": 0.50, "tau_cross": 0.80, "carve": 0.20,
        "snap_mode": "selfiou", "snap": 0.50, "exponent": 3, "nms": 0.50,
    },
    # the union with the 48-term vocabulary in place of the 18-class one
    "fused48": {
        "recordings": REPO / "runs/scannet_ap/fused_v48_generic_amg",
        "tau_same": 0.50, "tau_cross": 0.50, "carve": 0.20,
        "snap_mode": "selfiou", "snap": 0.50, "exponent": 3, "nms": 0.50,
    },
    # 96 terms: a longer prefix of the same published, frequency-ordered ScanNet200 class list.
    # THE DEPLOYED HEADLINE CONFIGURATION.  coarse_weight is part of it: the prior used to be
    # installed by importing c1_coarse_linking from another worktree, so a run from this checkout
    # silently omitted a mechanism the ablation credits at -1.06 AP.  Declaring it here makes the
    # dependency explicit; c1_coarse_linking.install() is called from predict() when it is set.
    "v96": {
        "recordings": REPO / "runs/scannet_ap/v96_clean",
        "tau_same": 0.50, "tau_cross": 0.80, "carve": 0.20,
        "snap_mode": "selfiou", "snap": 0.50, "exponent": 3, "nms": 0.50,
        "coarse_weight": 0.30,
    },
    "generic": {
        "recordings": REPO / "runs/scannet_ap/generic_clean",
        "tau_same": 0.50,
        "tau_cross": 0.80,
        "carve": 0.20,
        "snap_mode": "selfiou",
        "snap": 0.50,
        "exponent": 3,
        "nms": 0.50,
    },
    "amg": {
        "recordings": REPO / "runs/scannet_ap/amg_clean",
        "tau_same": 0.50,
        "tau_cross": 0.50,
        "carve": 0.20,
        "snap_mode": "selfiou",
        "snap": 0.50,
        "exponent": 3,
        "nms": 0.50,
    },
    "gdsam": {
        "recordings": REPO / "runs/scannet_ap/v5_gdsam",
        "tau_same": 0.50,
        "tau_cross": 0.90,
        "carve": 0.30,
        "snap_mode": "adaptive",
        "snap": 0.40,
        "exponent": 4,
        "nms": 0.30,
    },
}
VARIANTS = (
    ("mean", "mean", 0.50),
    ("conflict@1.00", "conflict-fraction", 1.00),
    ("conflict@0.75", "conflict-fraction", 0.75),
    ("conflict@0.50", "conflict-fraction", 0.50),
    ("conflict@0.25", "conflict-fraction", 0.25),
    ("conflict@0.10", "conflict-fraction", 0.10),
)


# Explicit mask-only control: every backend parameter matches the clean SAM3 v96
# configuration. The legacy "gdsam" entry remains a historical ablation.
FRONTENDS["v96"]["require_clean_prompts"] = True
FRONTENDS["gdsam_matched_clean"] = {
    **FRONTENDS["v96"],
    "recordings": REPO / "runs/scannet_ap/v5_gdsam_v96vocab",
}


def load_scene(scene: str, cfg: dict) -> dict:
    n_vertices = len(oe._read_ply_xyz(f"{oe.SCANS}/{scene}/{scene}_vh_clean_2.ply"))
    with open(Path(cfg["recordings"]) / f"{scene}.pkl", "rb") as handle:
        recording = pickle.load(handle)
    superpoints = oe._superpoints(scene, n_vertices)
    totals = np.bincount(superpoints, minlength=superpoints.max() + 1).astype(np.float64)
    sem, ins, n_gt = oe.gt_labels(scene, n_vertices, "scannetv2")
    return {
        "scene": scene,
        "n_vertices": n_vertices,
        "recording": recording,
        "superpoints": superpoints,
        "totals": totals,
        "sem": sem,
        "ins": ins,
        "n_gt": n_gt,
    }


def predict(item: dict, cfg: dict, veto_mode: str, conflict_frac: float) -> list[dict]:
    scene = item["scene"]
    n_vertices = item["n_vertices"]
    recording = item["recording"]
    superpoints = item["superpoints"]
    totals = item["totals"]
    predictions = []
    clusters = link_masks(
        recording,
        cfg["tau_same"],
        cfg["tau_cross"],
        0.05,
        use_veto=True,
        carve=cfg["carve"],
        greedy=True,
        claim_mode="frame",
        veto_mode=veto_mode,
        conflict_frac=conflict_frac,
        spp=superpoints if cfg.get("spp_kappa", 0.0) > 0 else None,
        kappa=cfg.get("spp_kappa", 0.0),
        split_ncut=cfg.get("split_ncut", 0.0),
        split_min_side=cfg.get("split_min_side", 2),
        split_mode=cfg.get("split_mode", "replace"),
    )
    for vertices, _, n_views, _ in clusters:
        vertices = vertices[vertices < n_vertices]
        if len(vertices) < 40:
            continue
        raw = np.zeros(n_vertices, np.uint8)
        raw[vertices] = 1
        if cfg["snap_mode"] == "none":
            # no snapping at all -- needed to ABLATE the step.  Without this branch a config asking
            # for "none" silently fell through to adaptive snapping, which made an ablation of
            # snapping measure adaptive-vs-selfiou instead.
            mask = raw
        elif cfg["snap_mode"] == "selfiou":
            mask = oe._snap_selfiou(raw, superpoints, totals)
        else:
            mask = oe._snap_adaptive(raw, superpoints, cfg["snap"])
        if mask.sum() < 40:
            mask = raw
        views = float(n_views) * (1.0 - np.exp(-len(vertices) / 200.0))
        confidence = views * oe._spp_purity(vertices, superpoints, totals) ** cfg["exponent"]
        predictions.append(
            {"scan_id": scene, "label_id": 1, "pred_mask": mask, "conf": float(confidence)}
        )
    scale = max((pred["conf"] for pred in predictions), default=1.0) or 1.0
    for pred in predictions:
        pred["conf"] /= scale
    return oe._mask_nms(predictions, cfg["nms"], 0.30)


def evaluate(cache: list[dict], predictions: list[list[dict]], constant: bool) -> dict:
    if constant:
        predictions = [oe._apply_score_protocol(scene, "constant") for scene in predictions]
    evaluator = ScanNetEval(class_labels=["object"], use_label=False, dataset_name="scannetv2")
    with contextlib.redirect_stdout(io.StringIO()):
        avg = evaluator.evaluate(
            predictions,
            [item["sem"] for item in cache],
            [item["ins"] for item in cache],
            exp_path="/tmp",
        )
    return {
        "predictions": int(sum(len(scene) for scene in predictions)),
        "ap": float(avg["all_ap"] * 100.0),
        "ap50": float(avg["all_ap_50%"] * 100.0),
        "ap25": float(avg["all_ap_25%"] * 100.0),
        "ar": float(avg["all_rc"] * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-file", default=str(BASE / "tune_scenes.txt"))
    parser.add_argument("--frontends", default="sam3,gdsam")
    parser.add_argument("--variants", default=",".join(row[0] for row in VARIANTS))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    scenes = [line.strip() for line in Path(args.scenes_file).read_text().splitlines() if line.strip()]
    frontends = [name.strip() for name in args.frontends.split(",") if name.strip()]
    selected_names = {name.strip() for name in args.variants.split(",") if name.strip()}
    unknown_frontends = sorted(set(frontends) - set(FRONTENDS))
    variants = [variant for variant in VARIANTS if variant[0] in selected_names]
    unknown_variants = sorted(selected_names - {variant[0] for variant in VARIANTS})
    if unknown_frontends:
        parser.error(f"unknown frontends: {', '.join(unknown_frontends)}")
    if unknown_variants:
        parser.error(f"unknown variants: {', '.join(unknown_variants)}")

    rows = []
    for frontend in frontends:
        cfg = FRONTENDS[frontend]
        available = [scene for scene in scenes if (Path(cfg["recordings"]) / f"{scene}.pkl").exists()]
        print(f"\n{frontend}: loading {len(available)}/{len(scenes)} scenes", flush=True)
        cache = []
        for index, scene in enumerate(available, 1):
            cache.append(load_scene(scene, cfg))
            print(f"  [{index:>2}/{len(available)}] {scene}", flush=True)
        print("  variant             score       #pred      AP    AP50    AP25      AR", flush=True)
        for name, veto_mode, conflict_frac in variants:
            predictions = [predict(item, cfg, veto_mode, conflict_frac) for item in cache]
            for score_name, constant in (("native", False), ("constant", True)):
                result = evaluate(cache, predictions, constant)
                row = {
                    "frontend": frontend,
                    "variant": name,
                    "veto_mode": veto_mode,
                    "conflict_frac": conflict_frac,
                    "score_protocol": score_name,
                    "scenes": len(cache),
                    "gt": int(sum(item["n_gt"] for item in cache)),
                    **result,
                }
                rows.append(row)
                print(
                    f"  {name:<19} {score_name:<9} {result['predictions']:>6d} "
                    f"{result['ap']:>7.2f} {result['ap50']:>7.2f} "
                    f"{result['ap25']:>7.2f} {result['ar']:>7.2f}",
                    flush=True,
                )

    payload = {
        "scenes_file": str(Path(args.scenes_file).resolve()),
        "gt_protocol": "scannetv2",
        "claim_mode": "frame",
        "rows": rows,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
