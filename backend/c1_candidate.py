#!/usr/bin/env python
"""Frozen Component 1 candidate: baseline-gated temporal/projection fusion.

This module contains the promoted inference path only.  It does not load annotations.  Research
sweeps and alternative settings remain in the ``c1_*_ablation.py`` files on the parent research
branch.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

import official_eval as oe
from c1_bridge_ablation import FRONTENDS, predict
from c1_projection_consistency import projection_consistency, rescore_and_select
from c1_temporal_tracking import consolidate_tracks, temporal_tracks, tracks_to_predictions


# Frozen on ScanNetV2 TUNE-8 across SAM3 and Grounded-SAM, then validated unchanged on DEV-16 and
# on both ScanNetV2 and ScanNet200 annotations.
NATIVE_FLOOR = 0.003
TRACK_QUEUE = 5
TRACK_THRESHOLD = 0.20
CONSOLIDATE_THRESHOLD = 0.60
CONSOLIDATE_COOCCUR = 0.50
PROJECTION_TOP_K = 20
TEMPORAL_PROJECTION_FLOOR = 0.70
FUSION_NMS = 0.50

# Follow-on proposal-conditioned re-lifting.  These two values were selected jointly across both
# frontends and annotation mappings on TUNE, then validated unchanged on DEV.
RELIFT_MATCH_FLOOR = 0.50
RELIFT_VOTE_FLOOR = 0.30


def _inference_item(scene: str, n_vertices: int, cfg: dict, recordings: Path,
                    require_clean_prompts: bool) -> dict | None:
    recording_path = recordings / f"{scene}.pkl"
    if not recording_path.exists():
        return None
    with recording_path.open("rb") as handle:
        recording = pickle.load(handle)
    if require_clean_prompts and recording.get("oracle_scene_prompts") is not False:
        raise RuntimeError(
            f"{recording_path} is not explicitly marked oracle-free; regenerate it with a fixed "
            "scene-independent vocabulary before a clean-protocol claim"
        )
    superpoints = oe._superpoints(scene, n_vertices)
    totals = np.bincount(superpoints, minlength=superpoints.max() + 1).astype(np.float64)
    return {
        "scene": scene,
        "n_vertices": n_vertices,
        "recording": recording,
        "superpoints": superpoints,
        "totals": totals,
    }


def _deployed_predictions(item: dict, cfg: dict) -> list[dict]:
    return [
        {**prediction, "proposal_path": "deployed"}
        for prediction in predict(item, cfg, "mean", 0.50)
        if prediction["conf"] >= NATIVE_FLOOR
    ]


def predict_candidate_item(item: dict, cfg: dict, return_deployed: bool = False):
    """Run temporal/projection fusion on an annotation-free in-memory inference item."""
    deployed = _deployed_predictions(item, cfg)
    tracks = temporal_tracks(
        item["recording"], TRACK_QUEUE, TRACK_THRESHOLD, use_concepts=False
    )
    tracks = consolidate_tracks(
        item["recording"],
        tracks,
        item["n_vertices"],
        CONSOLIDATE_THRESHOLD,
        "iou",
        CONSOLIDATE_COOCCUR,
    )
    temporal = [
        {**prediction, "proposal_path": "temporal"}
        for prediction in tracks_to_predictions(item, cfg, tracks, NATIVE_FLOOR)
    ]
    proposals = deployed + temporal
    scores = projection_consistency(item, proposals, PROJECTION_TOP_K, "geometric")
    candidate = rescore_and_select(
        proposals,
        scores,
        TEMPORAL_PROJECTION_FLOOR,
        FUSION_NMS,
        preserve_paths=frozenset({"deployed"}),
    )
    # Keep evaluator-facing records minimal and prevent downstream code from accidentally treating
    # proposal provenance as semantic information.
    for prediction in candidate:
        prediction.pop("proposal_path", None)
    for prediction in deployed:
        prediction.pop("proposal_path", None)
    return (deployed, candidate) if return_deployed else candidate


def predict_candidate(scene: str, n_vertices: int, frontend: str = "sam3",
                      recordings: str | os.PathLike | None = None,
                      require_clean_prompts: bool = True,
                      return_deployed: bool = False):
    """Run the frozen temporal/projection candidate without consulting ground truth.

    ``require_clean_prompts`` should be enabled for reportable end-to-end runs.  It intentionally
    rejects legacy recordings that have no provenance metadata, including this workspace's cached
    validation recordings.
    """
    if frontend not in FRONTENDS:
        raise ValueError(f"unknown frontend: {frontend}")
    cfg = dict(FRONTENDS[frontend])
    recording_dir = Path(recordings) if recordings is not None else Path(cfg["recordings"])
    item = _inference_item(scene, n_vertices, cfg, recording_dir, require_clean_prompts)
    if item is None:
        return ([], []) if return_deployed else []
    return predict_candidate_item(item, cfg, return_deployed=return_deployed)


def predict_relift_candidate(scene: str, n_vertices: int, frontend: str = "sam3",
                             recordings: str | os.PathLike | None = None,
                             require_clean_prompts: bool = True,
                             return_candidate: bool = False):
    """Run the frozen proposal-conditioned re-lifting pass over the temporal candidate."""
    if frontend not in FRONTENDS:
        raise ValueError(f"unknown frontend: {frontend}")
    cfg = dict(FRONTENDS[frontend])
    recording_dir = Path(recordings) if recordings is not None else Path(cfg["recordings"])
    item = _inference_item(scene, n_vertices, cfg, recording_dir, require_clean_prompts)
    if item is None:
        return ([], []) if return_candidate else []
    candidate = predict_candidate_item(item, cfg)
    # Import lazily: the ablation harness imports this module for the base-candidate adapter.
    from c1_relift_refinement import refine_candidate

    relifted, _ = refine_candidate(
        item,
        cfg,
        candidate,
        PROJECTION_TOP_K,
        RELIFT_MATCH_FLOOR,
        RELIFT_VOTE_FLOOR,
        TEMPORAL_PROJECTION_FLOOR,
        FUSION_NMS,
    )
    return (candidate, relifted) if return_candidate else relifted


def predict_deployed(scene: str, n_vertices: int, frontend: str = "sam3",
                     recordings: str | os.PathLike | None = None,
                     require_clean_prompts: bool = True) -> list[dict]:
    """Run only the frontend-matched, pruned deployed path for an efficient baseline."""
    if frontend not in FRONTENDS:
        raise ValueError(f"unknown frontend: {frontend}")
    cfg = dict(FRONTENDS[frontend])
    recording_dir = Path(recordings) if recordings is not None else Path(cfg["recordings"])
    item = _inference_item(scene, n_vertices, cfg, recording_dir, require_clean_prompts)
    if item is None:
        return []
    deployed = _deployed_predictions(item, cfg)
    for prediction in deployed:
        prediction.pop("proposal_path", None)
    return deployed


def predict_from_environment(scene: str, n_vertices: int):
    """Adapter used by ``official_eval.py`` and ``scannet200_eval.py``."""
    frontend = os.environ.get("C1_FRONTEND", "sam3")
    recordings = os.environ.get("HELD_DIR")
    require_clean = os.environ.get("C1_REQUIRE_CLEAN_PROMPTS", "1") == "1"
    return predict_candidate(
        scene,
        n_vertices,
        frontend=frontend,
        recordings=recordings,
        require_clean_prompts=require_clean,
    )


def deployed_from_environment(scene: str, n_vertices: int):
    """Frontend-matched, 0.003-pruned baseline used for candidate deltas."""
    frontend = os.environ.get("C1_FRONTEND", "sam3")
    recordings = os.environ.get("HELD_DIR")
    require_clean = os.environ.get("C1_REQUIRE_CLEAN_PROMPTS", "1") == "1"
    return predict_deployed(
        scene,
        n_vertices,
        frontend=frontend,
        recordings=recordings,
        require_clean_prompts=require_clean,
    )


def relift_from_environment(scene: str, n_vertices: int):
    """Adapter for the frozen proposal-conditioned re-lifting candidate."""
    frontend = os.environ.get("C1_FRONTEND", "sam3")
    recordings = os.environ.get("HELD_DIR")
    require_clean = os.environ.get("C1_REQUIRE_CLEAN_PROMPTS", "1") == "1"
    return predict_relift_candidate(
        scene,
        n_vertices,
        frontend=frontend,
        recordings=recordings,
        require_clean_prompts=require_clean,
    )
