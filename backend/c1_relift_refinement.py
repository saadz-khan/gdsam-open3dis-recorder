#!/usr/bin/env python
"""Proposal-conditioned multi-view re-lifting for Component 1.

The temporal/projection candidate scores a final 3D mask against the best frontend mask in each
visible frame, but it does not feed those correspondences back into the mask.  This experiment closes
that loop without labels or frontend-specific scores:

1. project a candidate proposal into its most-visible views;
2. select the frontend mask with maximum symmetric coverage in each view;
3. re-lift only those matched masks and retain vertices supported in enough selected views;
4. snap the reconstructed mask to geometric superpoints; and
5. let projection consistency choose between the original and reconstructed hypotheses.

This is a proposal-conditioned analogue of region refinement and 3D-guided mask matching.  It uses
only the recording contract (mask incidence, depth visibility, frame ids, and superpoints), ignores
concept identity, and never reads ground truth during prediction.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
from pathlib import Path

import numpy as np

import official_eval as oe
from c1_bridge_ablation import FRONTENDS, load_scene
from c1_candidate import predict_candidate
from c1_projection_consistency import projection_consistency, rescore_and_select
from v6_cluster import _consensus_ratio


def _parse_floats(spec: str) -> list[float]:
    return [float(value) for value in spec.split(",") if value.strip()]


def _check_recording_provenance(recording: dict, path: Path, allow_legacy_oracle: bool) -> None:
    if not allow_legacy_oracle and recording.get("oracle_scene_prompts") is not False:
        raise RuntimeError(
            f"{path} is not explicitly marked oracle-free; pass "
            "--allow-legacy-oracle-recordings only for a labelled backend ablation"
        )


def matched_masks_for_proposal(recording: dict, proposal_mask: np.ndarray, top_k: int,
                               match_floor: float) -> np.ndarray:
    """Select at most one symmetrically matching frontend mask per visible frame."""
    point_masks = recording["P"].tocsc()
    visibility = recording["Vis"].tocsc()
    mask_frame = np.asarray(recording["mask_frame"], np.int32)
    vertices = np.flatnonzero(proposal_mask)
    vertices = vertices[vertices < point_masks.shape[0]]
    if vertices.size == 0:
        return np.empty(0, np.int32)

    intersection = np.asarray(point_masks[vertices].sum(0)).ravel()
    proposal_visible = np.asarray(visibility[vertices].sum(0)).ravel()
    mask_size = np.asarray(point_masks.sum(0)).ravel()
    frame_order = np.argsort(-proposal_visible, kind="stable")
    frame_order = frame_order[proposal_visible[frame_order] > 0]
    selected: list[int] = []
    for frame in frame_order[:top_k]:
        candidates = np.where(mask_frame == frame)[0]
        if candidates.size == 0:
            continue
        proposal_coverage = intersection[candidates] / max(float(proposal_visible[frame]), 1.0)
        mask_coverage = intersection[candidates] / np.maximum(mask_size[candidates], 1.0)
        agreement = np.sqrt(proposal_coverage * mask_coverage)
        best_local = int(np.argmax(agreement))
        if float(agreement[best_local]) >= match_floor:
            selected.append(int(candidates[best_local]))
    return np.asarray(selected, np.int32)


def relift_prediction(item: dict, cfg: dict, prediction: dict, top_k: int,
                      match_floor: float, vote_floor: float) -> dict | None:
    """Reconstruct one proposal from its best per-view frontend correspondences."""
    recording = item["recording"]
    members = matched_masks_for_proposal(
        recording, prediction["pred_mask"], top_k, match_floor
    )
    if members.size < 2:
        return None

    point_masks = recording["P"].tocsc()
    visibility = recording["Vis"].tocsc()
    mask_frame = np.asarray(recording["mask_frame"], np.int32)
    vertices = np.unique(point_masks[:, members].indices)
    if vertices.size < 40:
        return None
    support = _consensus_ratio(
        point_masks,
        visibility,
        vertices,
        members,
        mask_frame,
        claim_mode="frame",
    )
    supported = vertices[support >= vote_floor]
    if supported.size >= 40:
        vertices = supported
    vertices = vertices[vertices < item["n_vertices"]]
    if vertices.size < 40:
        return None

    raw = np.zeros(item["n_vertices"], np.uint8)
    raw[vertices] = 1
    if cfg["snap_mode"] == "selfiou":
        mask = oe._snap_selfiou(raw, item["superpoints"], item["totals"])
    else:
        mask = oe._snap_adaptive(raw, item["superpoints"], cfg["snap"])
    if mask.sum() < 40:
        mask = raw
    return {
        "scan_id": item["scene"],
        "label_id": 1,
        "pred_mask": mask,
        "conf": 1.0,
        "proposal_path": "relift",
    }


def refine_candidate(item: dict, cfg: dict, candidate: list[dict], top_k: int,
                     match_floor: float, vote_floor: float, projection_floor: float,
                     nms: float) -> tuple[list[dict], int]:
    """Fuse proposal-conditioned reconstructions with an unchanged candidate fallback."""
    originals = [{**prediction, "proposal_path": "candidate"} for prediction in candidate]
    refinements = []
    for prediction in originals:
        refined = relift_prediction(
            item, cfg, prediction, top_k, match_floor, vote_floor
        )
        if refined is not None:
            refinements.append(refined)
    proposals = originals + refinements
    scores = projection_consistency(item, proposals, top_k, "geometric")
    output = rescore_and_select(
        proposals,
        scores,
        projection_floor,
        nms,
        preserve_paths=frozenset({"candidate"}),
    )
    for prediction in output:
        prediction.pop("proposal_path", None)
    return output, len(refinements)


def _evaluate(cache: list[dict], predictions: list[list[dict]], annotation: str) -> dict:
    from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval

    evaluator = ScanNetEval(
        class_labels=["object"],
        use_label=False,
        dataset_name="scannet200" if annotation == "scannet200" else "scannetv2",
    )
    constant = [oe._apply_score_protocol(scene, "constant") for scene in predictions]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        average = evaluator.evaluate(
            constant,
            [item["sem"] for item in cache],
            [item["ins"] for item in cache],
            exp_path="/tmp",
        )
    return {
        "scenes": len(cache),
        "gt": int(sum(item["n_gt"] for item in cache)),
        "predictions": int(sum(len(scene) for scene in predictions)),
        "ap": float(average["all_ap"] * 100.0),
        "ap50": float(average["all_ap_50%"] * 100.0),
        "ap25": float(average["all_ap_25%"] * 100.0),
        "ar": float(average["all_rc"] * 100.0),
    }


def _cache_for_annotation(cache: list[dict], annotation: str) -> list[dict]:
    if annotation == "scannetv2":
        return cache
    from scannet200_eval import gt200

    converted = []
    for item in cache:
        ground_truth = gt200(item["scene"], item["n_vertices"])
        if ground_truth is None:
            continue
        changed = dict(item)
        changed["sem"], changed["ins"], changed["n_gt"] = ground_truth
        converted.append(changed)
    return converted


def _align_predictions(source_cache: list[dict], predictions: list[list[dict]],
                       target_cache: list[dict]) -> list[list[dict]]:
    """Align proposal lists if an annotation mapping omits unavailable scenes."""
    by_scene = {
        item["scene"]: scene_predictions
        for item, scene_predictions in zip(source_cache, predictions)
    }
    return [by_scene[item["scene"]] for item in target_cache]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-file", default="tune_scenes.txt")
    parser.add_argument("--frontends", default="sam3,gdsam")
    parser.add_argument("--annotations", default="scannetv2,scannet200")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--match-floors", default="0.3,0.5,0.7")
    parser.add_argument("--vote-floors", default="0.2,0.3,0.5")
    parser.add_argument("--projection-floor", type=float, default=0.70)
    parser.add_argument("--nms", type=float, default=0.50)
    parser.add_argument("--allow-legacy-oracle-recordings", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--dump", default=None, help="optional prediction/GT dump for auditing")
    args = parser.parse_args()

    scenes = [
        line.strip()
        for line in Path(args.scenes_file).read_text().splitlines()
        if line.strip()
    ]
    frontends = [value.strip() for value in args.frontends.split(",") if value.strip()]
    annotations = [value.strip() for value in args.annotations.split(",") if value.strip()]
    unknown_frontends = sorted(set(frontends) - set(FRONTENDS))
    unknown_annotations = sorted(set(annotations) - {"scannetv2", "scannet200"})
    if unknown_frontends:
        parser.error(f"unknown frontends: {', '.join(unknown_frontends)}")
    if unknown_annotations:
        parser.error(f"unknown annotations: {', '.join(unknown_annotations)}")
    match_floors = _parse_floats(args.match_floors)
    vote_floors = _parse_floats(args.vote_floors)

    rows = []
    dump_payload: dict = {"scenes_file": str(Path(args.scenes_file).resolve()), "frontends": {}}
    for frontend in frontends:
        cfg = dict(FRONTENDS[frontend])
        recordings = Path(cfg["recordings"])
        available = [scene for scene in scenes if (recordings / f"{scene}.pkl").exists()]
        cache: list[dict] = []
        candidates: list[list[dict]] = []
        print(f"\n{frontend}: loading {len(available)}/{len(scenes)} scenes", flush=True)
        for index, scene in enumerate(available, 1):
            item = load_scene(scene, cfg)
            recording_path = recordings / f"{scene}.pkl"
            _check_recording_provenance(
                item["recording"], recording_path, args.allow_legacy_oracle_recordings
            )
            candidate = predict_candidate(
                scene,
                item["n_vertices"],
                frontend=frontend,
                recordings=recordings,
                require_clean_prompts=not args.allow_legacy_oracle_recordings,
            )
            cache.append(item)
            candidates.append(candidate)
            print(
                f"  [{index:>2}/{len(available)}] {scene}: {len(candidate)} candidate proposals",
                flush=True,
            )

        annotation_caches = {
            annotation: _cache_for_annotation(cache, annotation) for annotation in annotations
        }
        for annotation, annotation_cache in annotation_caches.items():
            aligned_candidates = _align_predictions(cache, candidates, annotation_cache)
            baseline = _evaluate(annotation_cache, aligned_candidates, annotation)
            row = {
                "frontend": frontend,
                "annotation": annotation,
                "variant": "candidate",
                "match_floor": None,
                "vote_floor": None,
                "refinements": 0,
                **baseline,
            }
            rows.append(row)
            print(
                f"  {annotation:<10} candidate             #pred={baseline['predictions']:4d} "
                f"AP={baseline['ap']:6.2f} AR={baseline['ar']:6.2f}",
                flush=True,
            )

        frontend_dump = {"candidates": candidates, "variants": {}, "gts": {}}
        for annotation, annotation_cache in annotation_caches.items():
            frontend_dump["gts"][annotation] = {
                item["scene"]: (item["sem"], item["ins"], item["n_gt"])
                for item in annotation_cache
            }
        for match_floor in match_floors:
            for vote_floor in vote_floors:
                refined_predictions = []
                refinement_count = 0
                for item, candidate in zip(cache, candidates):
                    output, count = refine_candidate(
                        item,
                        cfg,
                        candidate,
                        args.top_k,
                        match_floor,
                        vote_floor,
                        args.projection_floor,
                        args.nms,
                    )
                    refined_predictions.append(output)
                    refinement_count += count
                variant = f"match{match_floor:.2f}_vote{vote_floor:.2f}"
                frontend_dump["variants"][variant] = refined_predictions
                for annotation, annotation_cache in annotation_caches.items():
                    aligned_predictions = _align_predictions(
                        cache, refined_predictions, annotation_cache
                    )
                    result = _evaluate(annotation_cache, aligned_predictions, annotation)
                    row = {
                        "frontend": frontend,
                        "annotation": annotation,
                        "variant": "relift",
                        "match_floor": match_floor,
                        "vote_floor": vote_floor,
                        "refinements": refinement_count,
                        **result,
                    }
                    rows.append(row)
                    print(
                        f"  {annotation:<10} m={match_floor:.2f} v={vote_floor:.2f} "
                        f"ref={refinement_count:4d} #pred={result['predictions']:4d} "
                        f"AP={result['ap']:6.2f} AR={result['ar']:6.2f}",
                        flush=True,
                    )
        dump_payload["frontends"][frontend] = frontend_dump

    payload = {
        "scenes_file": str(Path(args.scenes_file).resolve()),
        "frontends": frontends,
        "annotations": annotations,
        "top_k": args.top_k,
        "projection_floor": args.projection_floor,
        "nms": args.nms,
        "allow_legacy_oracle_recordings": args.allow_legacy_oracle_recordings,
        "rows": rows,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {args.json_out}", flush=True)
    if args.dump:
        with Path(args.dump).open("wb") as handle:
            pickle.dump(dump_payload, handle)
        print(f"Wrote {args.dump}", flush=True)


if __name__ == "__main__":
    main()
