#!/usr/bin/env python
"""Score and fuse 3D proposals by label-free multi-view projection consistency.

For each 3D proposal, select the K frames in which the proposal has the most depth-visible vertices.
In every selected frame, find the frontend mask with maximum geometric-mean agreement:

    sqrt( fraction of visible proposal explained by the 2D mask
          * fraction of the 2D mask explained by the proposal )

The final score is the mean over views.  It is symmetric, scale-free, class-agnostic, and uses only
the same RGB-D projection evidence that produced the proposal.  In contrast to the existing support
score, it evaluates the *final snapped mask*, so it can choose between alternative proposal paths and
snap outcomes.  This is a points-only analogue of GVC-Seg's geometric-visual correspondence score,
but it needs neither a supervised 3D backbone nor a detector-specific confidence.

This file is an ablation harness.  It can score the deployed path alone or fuse it with the
class-agnostic temporal path from ``c1_temporal_tracking.py``.  All reported constant-score rows set
the surviving predictions to confidence 1.0 after selection, matching MV3DIS's class-agnostic
protocol.
"""
from __future__ import annotations

import argparse
import json
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
from c1_bridge_ablation import FRONTENDS, evaluate, load_scene, predict  # noqa: E402
from c1_temporal_tracking import (  # noqa: E402
    consolidate_tracks,
    temporal_tracks,
    tracks_to_predictions,
)


def projection_consistency(item: dict, predictions: list[dict], top_k: int,
                           metric: str = "geometric") -> np.ndarray:
    """Compute final-mask agreement with the best frontend mask in each top-visible view."""
    if metric not in {"geometric", "iou", "proposal", "minimum"}:
        raise ValueError(f"unknown projection metric: {metric}")
    recording = item["recording"]
    points = recording["P"].tocsc()
    visible = recording["Vis"].tocsc()
    mask_frame = np.asarray(recording["mask_frame"], np.int32)
    mask_size = np.asarray(points.sum(0)).ravel()
    scores = np.zeros(len(predictions), np.float64)
    for index, prediction in enumerate(predictions):
        vertices = np.flatnonzero(prediction["pred_mask"])
        vertices = vertices[vertices < points.shape[0]]
        if vertices.size == 0:
            continue
        intersection = np.asarray(points[vertices].sum(0)).ravel()
        proposal_visible = np.asarray(visible[vertices].sum(0)).ravel()
        frame_order = np.argsort(-proposal_visible)
        frame_order = frame_order[proposal_visible[frame_order] > 0]
        agreements = []
        for frame in frame_order[:top_k]:
            masks = np.where(mask_frame == frame)[0]
            if masks.size == 0:
                continue
            inter = intersection[masks]
            proposal_fraction = inter / max(float(proposal_visible[frame]), 1.0)
            mask_fraction = inter / np.maximum(mask_size[masks], 1.0)
            if metric == "geometric":
                agreement = np.sqrt(proposal_fraction * mask_fraction)
            elif metric == "iou":
                union = proposal_visible[frame] + mask_size[masks] - inter
                agreement = inter / np.maximum(union, 1.0)
            elif metric == "proposal":
                agreement = proposal_fraction
            else:
                agreement = np.minimum(proposal_fraction, mask_fraction)
            agreements.append(float(agreement.max()))
        scores[index] = float(np.mean(agreements)) if agreements else 0.0
    return scores


def rescore_and_select(predictions: list[dict], scores: np.ndarray, score_floor: float,
                       nms: float, preserve_paths: frozenset[str] = frozenset()) -> list[dict]:
    rescored = []
    for prediction, score in zip(predictions, scores):
        if score < score_floor and prediction.get("proposal_path") not in preserve_paths:
            continue
        item = prediction.copy()
        item["conf"] = float(score)
        rescored.append(item)
    return oe._mask_nms(rescored, nms, 0.30)


def parse_numbers(spec: str, cast=float) -> list:
    return [cast(value) for value in spec.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-file", default=str(BASE / "tune_scenes.txt"))
    parser.add_argument("--frontends", default="sam3,gdsam")
    parser.add_argument("--annotation", choices=("scannetv2", "scannet200"), default="scannetv2")
    parser.add_argument("--paths", default="deployed,fusion",
                        help="comma-separated subset of deployed,temporal,fusion")
    parser.add_argument("--top-k", default="10,20")
    parser.add_argument("--metric", choices=("geometric", "iou", "proposal", "minimum"),
                        default="geometric")
    parser.add_argument("--score-floors", default="0,0.4,0.5,0.6,0.7,0.75,0.8")
    parser.add_argument("--native-floor", type=float, default=0.003)
    parser.add_argument("--nms", type=float, default=0.50)
    parser.add_argument(
        "--fusion-floor-mode",
        choices=("temporal-only", "all"),
        default="temporal-only",
        help=("apply the projection floor only to added temporal proposals, preserving the deployed "
              "path through the floor (cross-path NMS may still replace overlaps); 'all' reproduces "
              "the original ablation"),
    )
    parser.add_argument("--track-queue", type=int, default=5)
    parser.add_argument("--track-threshold", type=float, default=0.20)
    parser.add_argument("--consolidate-threshold", type=float, default=0.60)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    scenes = [line.strip() for line in Path(args.scenes_file).read_text().splitlines() if line.strip()]
    frontends = [name.strip() for name in args.frontends.split(",") if name.strip()]
    paths = [name.strip() for name in args.paths.split(",") if name.strip()]
    unknown_frontends = sorted(set(frontends) - set(FRONTENDS))
    unknown_paths = sorted(set(paths) - {"deployed", "temporal", "fusion"})
    if unknown_frontends:
        parser.error(f"unknown frontends: {', '.join(unknown_frontends)}")
    if unknown_paths:
        parser.error(f"unknown proposal paths: {', '.join(unknown_paths)}")
    top_ks = parse_numbers(args.top_k, int)
    score_floors = parse_numbers(args.score_floors)
    rows = []

    for frontend in frontends:
        cfg = FRONTENDS[frontend]
        available = [
            scene for scene in scenes if (Path(cfg["recordings"]) / f"{scene}.pkl").exists()
        ]
        cache = [load_scene(scene, cfg) for scene in available]
        if args.annotation == "scannet200":
            from scannet200_eval import gt200

            converted = []
            for item in cache:
                ground_truth = gt200(item["scene"], item["n_vertices"])
                if ground_truth is None:
                    continue
                item = dict(item)
                item["sem"], item["ins"], item["n_gt"] = ground_truth
                converted.append(item)
            cache = converted
        print(f"\n{frontend}: building proposal paths for {len(cache)}/{len(scenes)} scenes", flush=True)
        proposals: dict[str, list[list[dict]]] = {name: [] for name in paths}
        for index, item in enumerate(cache, 1):
            deployed = [
                {**prediction, "proposal_path": "deployed"}
                for prediction in predict(item, cfg, "mean", 0.50)
                if prediction["conf"] >= args.native_floor
            ]
            temporal = []
            if any(name in paths for name in ("temporal", "fusion")):
                tracks = temporal_tracks(
                    item["recording"], args.track_queue, args.track_threshold, use_concepts=False
                )
                tracks = consolidate_tracks(
                    item["recording"], tracks, item["n_vertices"],
                    args.consolidate_threshold, "iou", 0.50,
                )
                temporal = [
                    {**prediction, "proposal_path": "temporal"}
                    for prediction in tracks_to_predictions(item, cfg, tracks, args.native_floor)
                ]
            if "deployed" in paths:
                proposals["deployed"].append(deployed)
            if "temporal" in paths:
                proposals["temporal"].append(temporal)
            if "fusion" in paths:
                proposals["fusion"].append(deployed + temporal)
            print(
                f"  [{index:>2}/{len(cache)}] {item['scene']}: "
                f"deployed={len(deployed)} temporal={len(temporal)}",
                flush=True,
            )

        print("  path/k/floor          score       #pred      AP    AP50    AP25      AR", flush=True)
        for path in paths:
            for top_k in top_ks:
                score_cache = [
                    projection_consistency(item, scene_predictions, top_k, args.metric)
                    for item, scene_predictions in zip(cache, proposals[path])
                ]
                all_scores = np.concatenate(score_cache) if score_cache else np.empty(0)
                if all_scores.size:
                    print(
                        f"    {path}/k{top_k} score quantiles: "
                        + ", ".join(
                            f"{value:.3f}"
                            for value in np.quantile(all_scores, [0, .1, .25, .5, .75, .9, 1])
                        ),
                        flush=True,
                    )
                for floor in score_floors:
                    preserve_paths = (
                        frozenset({"deployed"})
                        if path == "fusion" and args.fusion_floor_mode == "temporal-only"
                        else frozenset()
                    )
                    selected = [
                        rescore_and_select(
                            scene_predictions, scores, floor, args.nms, preserve_paths
                        )
                        for scene_predictions, scores in zip(proposals[path], score_cache)
                    ]
                    name = f"{path}/k{top_k}/f{floor:.2f}"
                    for score_name, constant in (("projection", False), ("constant", True)):
                        result = evaluate(cache, selected, constant)
                        row = {
                            "frontend": frontend,
                            "path": path,
                            "top_k": top_k,
                            "projection_metric": args.metric,
                            "score_floor": floor,
                            "score_protocol": score_name,
                            **result,
                        }
                        rows.append(row)
                        print(
                            f"  {name:<22} {score_name:<10} {result['predictions']:>6d} "
                            f"{result['ap']:>7.2f} {result['ap50']:>7.2f} "
                            f"{result['ap25']:>7.2f} {result['ar']:>7.2f}",
                            flush=True,
                        )

    payload = {
        "scenes_file": str(Path(args.scenes_file).resolve()),
        "annotation": args.annotation,
        "projection_metric": args.metric,
        "native_floor": args.native_floor,
        "nms": args.nms,
        "fusion_floor_mode": args.fusion_floor_mode,
        "track_queue": args.track_queue,
        "track_threshold": args.track_threshold,
        "consolidate_threshold": args.consolidate_threshold,
        "rows": rows,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
