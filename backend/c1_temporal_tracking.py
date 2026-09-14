#!/usr/bin/env python
"""Ablate class-agnostic temporal mask tracking before 3D consolidation.

This is a cached-data approximation of CDIS's first principle: do not treat every 2D mask as an
exchangeable observation.  Match masks over a short temporal queue using their depth-lifted support,
then reconnect tracks that do not co-occur but cover the same 3D object.  The implementation uses no
ground-truth labels and, by default, no prompt/concept identity.

The directional matching score is the IoU obtained after restricting a past mask to vertices visible
in the current frame.  This is the mesh-domain analogue of warping a past RGB-D mask into the current
view.  Hungarian assignment makes every track accept at most one mask per frame, preventing adjacent
same-class objects from being fused merely because both overlap a broad proposal.

This file is an ablation harness, not deployed code.  Select settings on TUNE and validate once on DEV.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

BASE = Path(__file__).resolve().parent
REPO = Path("/home/saad/Desktop/spacesculptor_old")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "Open3DIS"))

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import official_eval as oe  # noqa: E402
from c1_bridge_ablation import FRONTENDS, evaluate, load_scene, predict  # noqa: E402
from v6_cluster import _consensus_ratio  # noqa: E402


def _dominant_concept(members: list[int], concepts: np.ndarray) -> int:
    values = concepts[np.asarray(members, np.int64)]
    return int(np.bincount(values).argmax())


def temporal_tracks(recording: dict, queue: int, threshold: float,
                    use_concepts: bool = False) -> list[np.ndarray]:
    """Associate masks to short-lived tracks with one-to-one directional visible IoU."""
    n_masks = int(recording["nM"])
    if n_masks == 0:
        return []
    points = recording["P"].tocsc()
    visible = recording["Vis"].tocsc()
    mask_frame = np.asarray(recording["mask_frame"], np.int32)
    mask_concept = np.asarray(recording["mask_concept"], np.int32)
    overlap = (points.T @ points).toarray()
    mask_size = np.asarray(points.sum(0)).ravel()
    visible_size = (points.T @ visible).toarray()

    tracks: list[list[int]] = []
    last_frame: list[int] = []
    for frame in np.unique(mask_frame):
        current = np.where(mask_frame == frame)[0]
        active = [index for index, last in enumerate(last_frame) if last >= frame - queue]
        score = np.zeros((len(active), len(current)), np.float64)
        for row, track_index in enumerate(active):
            members = tracks[track_index]
            recent = np.asarray(
                [mask for mask in members if mask_frame[mask] >= frame - queue], np.int64
            )
            if recent.size == 0:
                continue
            inter = overlap[np.ix_(recent, current)]
            # Past mask warped into the current frame: only its currently visible support enters
            # the union.  The current mask already contains only support observed in this frame.
            union = visible_size[np.ix_(recent, [int(frame)])] + mask_size[current][None, :] - inter
            pair_score = inter / np.maximum(union, 1.0)
            if use_concepts:
                concept = _dominant_concept(members, mask_concept)
                pair_score[:, mask_concept[current] != concept] = 0.0
            score[row] = pair_score.max(0)

        assigned = set()
        if score.size:
            rows, cols = linear_sum_assignment(-score)
            for row, col in zip(rows.tolist(), cols.tolist()):
                if score[row, col] < threshold:
                    continue
                track_index = active[row]
                mask = int(current[col])
                tracks[track_index].append(mask)
                last_frame[track_index] = int(frame)
                assigned.add(mask)
        for mask in current:
            mask = int(mask)
            if mask in assigned:
                continue
            tracks.append([mask])
            last_frame.append(int(frame))
    return [np.asarray(track, np.int32) for track in tracks]


def _raw_track_incidence(recording: dict, tracks: list[np.ndarray],
                         n_vertices: int) -> sp.csc_matrix:
    """Return vertex-by-track incidence without materialising a huge dense mask matrix."""
    points = recording["P"][:n_vertices].tocsc()
    rows = np.concatenate(tracks) if tracks else np.empty(0, np.int32)
    cols = np.concatenate(
        [np.full(len(track), index, np.int32) for index, track in enumerate(tracks)]
    ) if tracks else np.empty(0, np.int32)
    membership = sp.csr_matrix(
        (np.ones(rows.size, np.float32), (rows, cols)),
        shape=(int(recording["nM"]), len(tracks)),
    )
    incidence = (points @ membership).tocsc()
    incidence.data[:] = 1.0
    return incidence


def consolidate_tracks(recording: dict, tracks: list[np.ndarray], n_vertices: int,
                       threshold: float, metric: str, cooccur_threshold: float) -> list[np.ndarray]:
    """Reconnect temporally disjoint track fragments using conservative 3D overlap.

    Tracks that often occur in the same frame are treated as cannot-link evidence.  This preserves
    neighboring objects while allowing an object that disappears and later reappears to reconnect.
    """
    if threshold <= 0 or len(tracks) < 2:
        return tracks
    if metric not in {"iou", "containment"}:
        raise ValueError(f"unknown consolidation metric: {metric}")
    mask_frame = np.asarray(recording["mask_frame"], np.int32)
    groups = [track.copy() for track in tracks]
    for _ in range(30):
        incidence = _raw_track_incidence(recording, groups, n_vertices)
        sizes = np.asarray(incidence.sum(0)).ravel().astype(np.float64)
        inter = (incidence.T @ incidence).toarray()
        if metric == "iou":
            similarity = inter / np.maximum(sizes[:, None] + sizes[None, :] - inter, 1.0)
        else:
            similarity = inter / np.maximum(np.minimum.outer(sizes, sizes), 1.0)
        np.fill_diagonal(similarity, -1.0)
        candidates = similarity >= threshold
        for left in range(len(groups)):
            frames_left = np.unique(mask_frame[groups[left]])
            for right in range(left + 1, len(groups)):
                if not candidates[left, right]:
                    continue
                frames_right = np.unique(mask_frame[groups[right]])
                cooccur = np.intersect1d(frames_left, frames_right).size / max(
                    min(frames_left.size, frames_right.size), 1
                )
                if cooccur > cooccur_threshold:
                    candidates[left, right] = candidates[right, left] = False
        if not candidates.any():
            break
        # Mutual-best matching avoids single-linkage chains while permitting many disjoint pairs per
        # round.  It is deterministic because track order and argmax tie-breaking are deterministic.
        score = np.where(candidates, similarity, -1.0)
        best = score.argmax(1)
        used = np.zeros(len(groups), bool)
        pairs = []
        for left, right in enumerate(best):
            right = int(right)
            if left == right or used[left] or used[right] or score[left, right] < 0:
                continue
            if best[right] != left:
                continue
            used[left] = used[right] = True
            pairs.append((left, right))
        if not pairs:
            break
        pair_by_left = {left: right for left, right in pairs}
        paired_right = {right for _, right in pairs}
        merged = []
        for index, group in enumerate(groups):
            if index in paired_right:
                continue
            if index in pair_by_left:
                group = np.unique(np.concatenate([group, groups[pair_by_left[index]]])).astype(np.int32)
            merged.append(group)
        groups = merged
    return groups


def tracks_to_predictions(item: dict, cfg: dict, tracks: list[np.ndarray], floor: float) -> list[dict]:
    recording = item["recording"]
    points = recording["P"].tocsc()
    visible = recording["Vis"].tocsc()
    mask_frame = np.asarray(recording["mask_frame"], np.int32)
    n_vertices = item["n_vertices"]
    superpoints = item["superpoints"]
    totals = item["totals"]
    predictions = []
    for members in tracks:
        vertices = np.unique(points[:, members].indices)
        if vertices.size < 40:
            continue
        ratio = _consensus_ratio(
            points, visible, vertices, members, mask_frame, claim_mode="frame"
        )
        keep = ratio >= cfg["carve"]
        if keep.sum() >= 40:
            vertices = vertices[keep]
        vertices = vertices[vertices < n_vertices]
        if vertices.size < 40:
            continue
        raw = np.zeros(n_vertices, np.uint8)
        raw[vertices] = 1
        if cfg["snap_mode"] == "selfiou":
            mask = oe._snap_selfiou(raw, superpoints, totals)
        else:
            mask = oe._snap_adaptive(raw, superpoints, cfg["snap"])
        if mask.sum() < 40:
            mask = raw
        n_views = np.unique(mask_frame[members]).size
        views = float(n_views) * (1.0 - np.exp(-vertices.size / 200.0))
        confidence = views * oe._spp_purity(vertices, superpoints, totals) ** cfg["exponent"]
        predictions.append(
            {
                "scan_id": item["scene"],
                "label_id": 1,
                "pred_mask": mask,
                "conf": float(confidence),
            }
        )
    scale = max((pred["conf"] for pred in predictions), default=1.0) or 1.0
    for pred in predictions:
        pred["conf"] /= scale
    predictions = [pred for pred in predictions if pred["conf"] >= floor]
    return oe._mask_nms(predictions, cfg["nms"], 0.30)


def parse_floats(spec: str) -> list[float]:
    return [float(value) for value in spec.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-file", default=str(BASE / "tune_scenes.txt"))
    parser.add_argument("--frontends", default="sam3,gdsam")
    parser.add_argument("--annotation", choices=("scannetv2", "scannet200"), default="scannetv2")
    parser.add_argument("--queue", type=int, default=5)
    parser.add_argument("--track-thresholds", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--consolidate-threshold", type=float, default=0.6)
    parser.add_argument("--consolidate-metric", choices=("iou", "containment"), default="iou")
    parser.add_argument("--cooccur-threshold", type=float, default=0.5)
    parser.add_argument("--use-concepts", action="store_true")
    parser.add_argument("--floor", type=float, default=0.003)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    scenes = [line.strip() for line in Path(args.scenes_file).read_text().splitlines() if line.strip()]
    frontends = [name.strip() for name in args.frontends.split(",") if name.strip()]
    unknown = sorted(set(frontends) - set(FRONTENDS))
    if unknown:
        parser.error(f"unknown frontends: {', '.join(unknown)}")
    thresholds = parse_floats(args.track_thresholds)
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
        print(f"\n{frontend}: {len(cache)}/{len(scenes)} scenes", flush=True)
        print("  variant             score       #pred      AP    AP50    AP25      AR", flush=True)
        baseline = []
        for item in cache:
            scene_predictions = predict(item, cfg, "mean", 0.50)
            scene_predictions = [p for p in scene_predictions if p["conf"] >= args.floor]
            baseline.append(scene_predictions)
        for score_name, constant in (("native", False), ("constant", True)):
            result = evaluate(cache, baseline, constant)
            rows.append(
                {
                    "frontend": frontend,
                    "variant": "deployed",
                    "score_protocol": score_name,
                    **result,
                }
            )
            print(
                f"  {'deployed':<19} {score_name:<9} {result['predictions']:>6d} "
                f"{result['ap']:>7.2f} {result['ap50']:>7.2f} {result['ap25']:>7.2f} "
                f"{result['ar']:>7.2f}",
                flush=True,
            )
        for threshold in thresholds:
            predictions = []
            track_counts = []
            merged_counts = []
            for index, item in enumerate(cache, 1):
                tracks = temporal_tracks(
                    item["recording"], args.queue, threshold, use_concepts=args.use_concepts
                )
                track_counts.append(len(tracks))
                tracks = consolidate_tracks(
                    item["recording"],
                    tracks,
                    item["n_vertices"],
                    args.consolidate_threshold,
                    args.consolidate_metric,
                    args.cooccur_threshold,
                )
                merged_counts.append(len(tracks))
                predictions.append(tracks_to_predictions(item, cfg, tracks, args.floor))
                print(
                    f"    [{index:>2}/{len(cache)}] t={threshold:.2f} {item['scene']}: "
                    f"{track_counts[-1]} -> {merged_counts[-1]} tracks",
                    flush=True,
                )
            name = f"track@{threshold:.2f}"
            for score_name, constant in (("native", False), ("constant", True)):
                result = evaluate(cache, predictions, constant)
                row = {
                    "frontend": frontend,
                    "variant": name,
                    "track_threshold": threshold,
                    "score_protocol": score_name,
                    "tracks_before": int(sum(track_counts)),
                    "tracks_after": int(sum(merged_counts)),
                    **result,
                }
                rows.append(row)
                print(
                    f"  {name:<19} {score_name:<9} {result['predictions']:>6d} "
                    f"{result['ap']:>7.2f} {result['ap50']:>7.2f} {result['ap25']:>7.2f} "
                    f"{result['ar']:>7.2f}",
                    flush=True,
                )

    payload = {
        "scenes_file": str(Path(args.scenes_file).resolve()),
        "annotation": args.annotation,
        "queue": args.queue,
        "consolidate_threshold": args.consolidate_threshold,
        "consolidate_metric": args.consolidate_metric,
        "cooccur_threshold": args.cooccur_threshold,
        "use_concepts": args.use_concepts,
        "floor": args.floor,
        "rows": rows,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
