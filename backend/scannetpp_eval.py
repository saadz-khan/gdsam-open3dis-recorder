#!/usr/bin/env python
"""Class-agnostic ScanNet++ v1 ground truth in the official evaluator's encoding.

PROTOCOL, SETTLED BY THE DATASET RATHER THAN BY PREFERENCE.  `top100_instance.txt` lists 83 classes
and deliberately excludes `wall`, `floor` and `ceiling`, all three of which ARE present in the
99-class semantic list `top100.txt`.  That exclusion is the benchmark's own definition of what counts
as an instance, so the class-agnostic task scores exactly those 83 classes.  This also matches how
the ScanNet task is defined here, where wall and floor are not instances either, which is what keeps
the two datasets comparable.

Two further rules, both verified against the data:
  * `SPLIT` appears as a segGroup label but is absent from `semantic_classes.txt` entirely.  It is an
    annotation-tool artifact, not an object, and is dropped rather than scored.
  * `segments.json` is the IDENTITY map -- one segment per vertex -- so a segGroup's `segments` list
    is already a list of vertex indices.  It supplies no over-segmentation.

Returns `(sem, ins, n_inst)` in the same encoding `official_eval.gt_labels` uses: a single foreground
class written as semantic id 2, instances 0-indexed, background -1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPP = Path("/home/saad/Desktop/scannetpp/data")
META = Path("/home/saad/Desktop/scannetpp/metadata")
MIN_REGION = 100                       # the evaluator's own minimum region size


def instance_classes() -> set[str]:
    """The 83 benchmark instance classes (structure deliberately excluded)."""
    return {l.strip() for l in open(META / "semantic_benchmark/top100_instance.txt") if l.strip()}


def gt_scannetpp(scene: str, nV: int, classes: set[str] | None = None):
    anno = SPP / scene / "scans/segments_anno.json"
    if not anno.exists():
        return None
    keep = instance_classes() if classes is None else classes
    groups = json.load(open(anno))["segGroups"]

    sem = np.zeros(nV, np.int64)
    ins = np.full(nV, -1, np.int64)
    k = 0
    for g in groups:
        label = str(g.get("label", "")).strip()
        if label not in keep:                       # excludes structure and the SPLIT artifact
            continue
        v = np.asarray(g["segments"], np.int64)
        v = v[(v >= 0) & (v < nV)]
        if v.size < MIN_REGION:
            continue
        sem[v] = 2                                  # single foreground class, evaluator convention
        ins[v] = k
        k += 1
    return sem, ins, k


def audit(scene: str, nV: int) -> dict:
    """Report what the class filter removes, so the protocol is inspectable rather than asserted."""
    groups = json.load(open(SPP / scene / "scans/segments_anno.json"))["segGroups"]
    keep = instance_classes()
    labels = [str(g.get("label", "")).strip() for g in groups]
    sizes = [len(g["segments"]) for g in groups]
    return {"annotated": len(groups),
            "kept": sum(1 for l, s in zip(labels, sizes) if l in keep and s >= MIN_REGION),
            "dropped_class": sorted({l for l in labels if l not in keep}),
            "dropped_small": sum(1 for l, s in zip(labels, sizes) if l in keep and s < MIN_REGION)}


# ---------------------------------------------------------------------------------------------
# Inference plumbing: ScanNet++ ships no over-segmentation, so we compute and cache our own.
# ---------------------------------------------------------------------------------------------
SPP_CACHE = Path("/home/saad/Desktop/spacesculptor_old/runs/scannetpp/superpoints")
FH_K, FH_MIN = 0.05, 20                # the ScanNet snap grid is ~216 verts/superpoint; this matches


def spp_superpoints(scene: str) -> np.ndarray:
    """Felzenszwalb over-segmentation of the aligned mesh, cached.

    ScanNet supplies official `.segs.json` superpoints and Steps 1 and 5 snap to them. ScanNet++
    supplies none -- `segments.json` is the identity map -- so this partition is OURS. That is a
    protocol difference between the two datasets and must be stated, not left implicit.
    """
    SPP_CACHE.mkdir(parents=True, exist_ok=True)
    out = SPP_CACHE / f"k{FH_K}_m{FH_MIN}" / f"{scene}.npy"
    if out.exists():
        return np.load(out)
    import mesh_oversegment as mo
    V, F = mo.read_ply_mesh(str(SPP / scene / "scans/mesh_aligned_0.05.ply"))
    lab = mo.oversegment(V, F, k=FH_K, min_verts=FH_MIN).astype(np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, lab)
    return lab


def spp_item(scene: str, recordings) -> dict | None:
    """The `_inference_item` contract, with our superpoints instead of the dataset's."""
    import pickle
    p = Path(recordings) / f"{scene}.pkl"
    if not p.exists():
        return None
    with p.open("rb") as h:
        recording = pickle.load(h)
    if recording.get("oracle_scene_prompts") is not False:
        raise RuntimeError(f"{p} is not marked oracle-free")
    sp = spp_superpoints(scene)
    n_vertices = int(recording["P"].shape[0])
    if len(sp) != n_vertices:
        raise ValueError(f"{scene}: {len(sp)} superpoint labels for {n_vertices} vertices")
    return {"scene": scene, "n_vertices": n_vertices, "recording": recording,
            "superpoints": sp,
            "totals": np.bincount(sp, minlength=sp.max() + 1).astype(np.float64)}
