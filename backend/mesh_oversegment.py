#!/usr/bin/env python
"""GEOMETRIC MESH OVER-SEGMENTATION — supplying superpoints for datasets that do not ship them.

ScanNet distributes `*_vh_clean_2.0.010000.segs.json`, produced by the Felzenszwalb-Huttenlocher
graph segmentation that ships with the ScanNet toolkit. Our pipeline leans on it heavily (removing the
snap costs 16.4 AP) and SAI3D consumes it too, so any dataset we want to compare on must have an
equivalent. Replica ships none.

Re-deriving it is straightforward -- the algorithm is published and parameterised by two numbers -- but
a *silently different* segmentation would make cross-dataset numbers meaningless: a coarser
over-segmentation flatters snapping, a finer one cripples it. So this module is written to be
VALIDATED, not trusted: `--validate` runs it on ScanNet scenes that already have an official
segmentation and reports how closely the two agree, and the downstream evaluation reports our method
under both. Replica numbers are only meaningful once that agreement is quantified.

Algorithm (Felzenszwalb & Huttenlocher 2004, as used by ScanNet):
  * graph over mesh vertices, one edge per mesh edge;
  * edge weight = 1 - <n_i, n_j>, i.e. normal dissimilarity, so segments break at creases;
  * merge components while  w <= min(Int(A) + k/|A|,  Int(B) + k/|B|);
  * absorb components smaller than `min_verts` into their lowest-weight neighbour.

  <env>/python mesh_oversegment.py --validate --scenes-file tune_scenes.txt
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
REPO = "/home/saad/Desktop/spacesculptor_old"
BASE = "/home/saad/Desktop/spacesculptor_baselines"
sys.path.insert(0, REPO); sys.path.insert(0, BASE)


class DSU:
    def __init__(self, n):
        self.p = np.arange(n); self.sz = np.ones(n, np.int64); self.int_ = np.zeros(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x

    def union(self, a, b, w):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a; self.sz[a] += self.sz[b]; self.int_[a] = w
        return True


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def oversegment(V, F, k=0.01, min_verts=20):
    """Felzenszwalb-Huttenlocher over the mesh graph with normal-dissimilarity weights."""
    N = vertex_normals(V, F)
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    w = 1.0 - np.einsum("ij,ij->i", N[E[:, 0]], N[E[:, 1]])
    order = np.argsort(w)
    E, w = E[order], w[order]

    d = DSU(len(V))
    for (a, b), ww in zip(E, w):
        ra, rb = d.find(a), d.find(b)
        if ra == rb:
            continue
        if ww <= min(d.int_[ra] + k / d.sz[ra], d.int_[rb] + k / d.sz[rb]):
            d.union(ra, rb, ww)
    # absorb undersized components into their lowest-weight neighbour
    for (a, b), ww in zip(E, w):
        ra, rb = d.find(a), d.find(b)
        if ra != rb and (d.sz[ra] < min_verts or d.sz[rb] < min_verts):
            d.union(ra, rb, ww)
    root = np.array([d.find(i) for i in range(len(V))])
    _, lab = np.unique(root, return_inverse=True)
    return lab.astype(np.int32)


def read_ply_mesh(path):
    from plyfile import PlyData
    p = PlyData.read(path)
    v = p["vertex"].data
    V = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    fe = p["face"].data
    key = "vertex_indices" if "vertex_indices" in fe.dtype.names else fe.dtype.names[0]
    # fan-triangulate any polygon arity; truncating quads to 3 indices drops an edge per face and
    # disconnects the mesh graph, which makes the segmentation meaningless (seen on Replica).
    tris = []
    for t in fe[key]:
        a = np.asarray(t, np.int64)
        tris.extend((a[0], a[j], a[j + 1]) for j in range(1, len(a) - 1))
    return V, np.asarray(tris, np.int64)


def agreement(a, b):
    """How well two over-segmentations agree: mean over segments of the best-overlap fraction."""
    out = []
    for s in np.unique(a):
        m = a == s
        vals, cnt = np.unique(b[m], return_counts=True)
        out.append(cnt.max() / m.sum())
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--scenes-file", default=f"{BASE}/tune_scenes.txt")
    ap.add_argument("--k", type=float, default=0.01)
    ap.add_argument("--min-verts", type=int, default=20)
    ap.add_argument("--limit", type=int, default=4)
    args = ap.parse_args()

    if not args.validate:
        print("nothing to do; use --validate or import oversegment()")
        return
    scenes = [s.strip() for s in open(args.scenes_file) if s.strip()][: args.limit]
    print(f"\n  === OVER-SEGMENTATION VALIDATION vs ScanNet's shipped segmentation "
          f"(k={args.k}, min_verts={args.min_verts}) ===")
    print(f"  {'scene':16s} {'#ours':>8s} {'#scannet':>9s} {'ours->sn':>9s} {'sn->ours':>9s}")
    for sc in scenes:
        sd = f"{REPO}/datasets/scannet_raw/scans/{sc}"
        V, F = read_ply_mesh(f"{sd}/{sc}_vh_clean_2.ply")
        ours = oversegment(V, F, args.k, args.min_verts)
        ref = np.asarray(json.load(open(f"{sd}/{sc}_vh_clean_2.0.010000.segs.json"))["segIndices"])
        _, ref = np.unique(ref[: len(V)], return_inverse=True)
        print(f"  {sc:16s} {ours.max()+1:8d} {ref.max()+1:9d} "
              f"{agreement(ours, ref):9.3f} {agreement(ref, ours):9.3f}", flush=True)
    print("\n  'ours->sn' = mean fraction of each of OUR segments falling in a single ScanNet segment")
    print("  (1.0 = ours is a strict refinement of ScanNet's); 'sn->ours' is the converse.")


if __name__ == "__main__":
    main()
