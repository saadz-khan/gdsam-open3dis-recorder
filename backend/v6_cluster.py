#!/usr/bin/env python
"""OURS v6 -- SEMANTICALLY-PRIMED, GEOMETRY-ANCHORED MASK LINKING (the Component-1 push).

Why v5 (a faithful MaskClustering reimplementation) under-merged 9x: MC links two masks only when a
single 2D mask CONTAINS both in the frames where both are visible. Masks of one object seen from
DIFFERENT viewpoints cover different surface patches, so mutual containment fails and the view-consensus
rate never reaches the 0.9 gate -> the object stays shattered.

v6 replaces that with the evidence MC cannot use, because MC is class-agnostic and 2D-only:
  (1) GEOMETRY: masks are linked by 3D overlap of their back-projected vertex sets. Different viewpoints
      of one object overlap partially, and transitivity (A-B, B-C => one object) closes the chain that
      pairwise 2D containment cannot.
  (2) SEMANTICS: our masks carry a concept. Same-concept pairs get the base threshold; cross-concept
      pairs must clear a stricter one (they are usually genuinely different objects).
  (3) VIEW-CONSENSUS VETO: MC's rate is kept, but as a NEGATIVE check -- if in the frames where both
      masks are visible a single 2D mask never contains both, they are separate objects (this is what
      keeps touching objects, e.g. chair-into-table, apart).
  MUTUAL-NEAREST-NEIGHBOUR agglomeration (not single-linkage) prevents the classic chaining blow-up,
  and a physical-size guard forbids merges producing implausibly large objects.

Then the validated v3 box-formation (cc_filter + support_trim) + NMS. Runs offline on v5_record.py data.

  PYTHONPATH=<repo> <sp311>/python v6_cluster.py [--sweep] [--write out.json]
"""
import argparse, glob, json, pickle, sys
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import DBSCAN
sys.path.insert(0, "/home/saad/Desktop/spacesculptor_old")
from scripts.scannet_io import load_gt_instances
from src.full_scene.eval.gt_metrics import obb_iou_3d
UP = np.array([0., 0., 1.])
V5 = "/home/saad/Desktop/spacesculptor_old/runs/scannet_ap/v5"
SCAN = "/home/saad/Desktop/spacesculptor_old/datasets/scannet_raw/scans"
SCENES = ["scene0011_00","scene0015_00","scene0019_00","scene0025_00",
          "scene0030_00","scene0046_00","scene0050_00","scene0063_00"]


# ---------- v3 box formation ----------
def sor(g, k=2.5):
    c=np.median(g,0); d=np.linalg.norm(g-c,axis=1); mad=np.median(np.abs(d-np.median(d)))*1.4826+1e-9
    gk=g[d<=np.median(d)+k*mad]; return gk if len(gk)>=20 else g
def cc_filter(g, frac=0.15, eps=0.10, mn=15):
    if len(g)<mn: return g
    cl=DBSCAN(eps=eps,min_samples=mn).fit_predict(g); thr=max(mn,frac*len(g)); keep=np.zeros(len(g),bool)
    for kk in set(cl)-{-1}:
        m=cl==kk
        if m.sum()>=thr: keep|=m
    return g[keep] if keep.sum()>=20 else g
def support_trim(g, band=0.03):
    z=g[:,2]; base=float(np.percentile(z,2)); body=g[z>base+band]
    if len(body)<20: return g
    blo,bhi=np.percentile(body[:,:2],2,0),np.percentile(body[:,:2],98,0)
    m=z>base+band; inside=np.all((g[:,:2]>=blo-0.05)&(g[:,:2]<=bhi+0.05),axis=1)
    keep=m|(~m&inside); return g[keep] if keep.sum()>=20 else g
def best_box(g):
    g=support_trim(cc_filter(sor(g))); return g, g.min(0), g.max(0)
def aiou(a,b):
    lo=np.maximum(a[0],b[0]); hi=np.minimum(a[1],b[1]); inter=np.prod(np.clip(hi-lo,0,None))
    return inter/(np.prod(a[1]-a[0])+np.prod(b[1]-b[0])-inter+1e-9)
def iou(c,s,gc,gs): return obb_iou_3d(c,s,0,gc,gs,0,up=UP)
def ofrac(c1,s1,c2,s2):
    lo=np.maximum(c1-s1/2,c2-s2/2); hi=np.minimum(c1+s1/2,c2+s2/2)
    inter=float(np.prod(np.clip(hi-lo,0,None))); den=min(float(np.prod(s1)),float(np.prod(s2)))
    return inter/den if den>0 else 0.0


def _consensus_ratio(P, Vis, vids, members, mask_frame, claim_mode="frame",
                     spp=None, kappa=0.0):
    """Return per-vertex claim/visibility evidence for one linked cluster.

    ``mask`` reproduces the original implementation: every member mask contributes one vote, even
    when several masks came from the same frame, while the denominator counts that frame once.
    ``frame`` caps the numerator at one vote per frame.  The latter is invariant to a frontend
    emitting duplicate or multi-granular masks in one image and guarantees a ratio in [0, 1].

    SEGMENT SHRINKAGE (``spp`` + ``kappa``).  The raw ratio is a per-vertex proportion estimated
    from ``seen(v)`` Bernoulli trials, and ``seen(v)`` is often three or four.  A hard threshold on
    a three-sample proportion is a coin flip: the measured consequence is that 99% of the instances
    we miss at IoU 0.50 are already TOUCHED, at a median coverage of 69%, and 61% of the surface we
    fail to claim WAS seen by the cameras.  The evidence was there and a noisy per-vertex test threw
    it away.

    Vertices are not independent, though -- they sit in geometric segments that are surfaces of one
    object almost by construction.  So pool within the segment and shrink each vertex toward its
    segment's rate, which is the Beta-Binomial posterior mean under a Beta prior centred on R(S)
    with strength kappa:

        r(v) = (claim(v) + kappa * R(S)) / (seen(v) + kappa),   R(S) = sum_S claim / sum_S seen

    kappa = 0 returns the current estimator EXACTLY (verified bit-exact in the probe); kappa -> inf
    admits or rejects whole segments, which is the partition-form behaviour the protocol rewards.
    One parameter interpolates between the two, so it can be selected rather than argued about.

    This is NOT the rejected v8 co-claim superpoint graph.  That built affinity EDGES between
    superpoints and clustered them -- a new grouping algorithm, which failed its held-out gate.
    This changes only the SUPPORT of an estimator that already exists, and adds no edges, no
    grouping, and no new decision.
    """
    if claim_mode not in {"mask", "frame"}:
        raise ValueError(f"unknown claim mode: {claim_mode}")
    frames, frame_inverse = np.unique(mask_frame[members], return_inverse=True)
    point_masks = P[vids][:, members]
    if claim_mode == "mask":
        claim = np.asarray(point_masks.sum(1)).ravel()
    else:
        mask_to_frame = sp.csr_matrix(
            (np.ones(len(members), np.float32), (np.arange(len(members)), frame_inverse)),
            shape=(len(members), len(frames)),
        )
        claimed_frames = (point_masks @ mask_to_frame).tocsr()
        claimed_frames.data[:] = 1.0
        claim = np.asarray(claimed_frames.sum(1)).ravel()
    seen = np.asarray(Vis[vids][:, frames].sum(1)).ravel()
    den = np.maximum(seen, 1).astype(np.float64)
    if spp is None or kappa <= 0:
        return claim / den                      # bit-exact original path
    # pool claim and visibility within each geometric segment, then shrink each vertex toward it.
    # vids can run past the mesh (predict() clips only afterwards); those get their own bucket, so
    # they are shrunk toward themselves and behave exactly as they do at kappa = 0.
    spp = np.asarray(spp)
    inside = vids < len(spp)
    sid = np.empty(len(vids), np.int64)
    sid[inside] = spp[vids[inside]]
    # each stray vertex becomes its OWN segment: a singleton pools with itself, and
    # (c + k*c/d) / (d + k) == c/d exactly, so those vertices are untouched by any kappa
    base = int(spp.max()) + 1 if len(spp) else 0
    sid[~inside] = base + np.arange(int((~inside).sum()))
    n = int(sid.max()) + 1 if len(sid) else 1
    seg_claim = np.bincount(sid, weights=claim, minlength=n)
    seg_seen = np.bincount(sid, weights=den, minlength=n)
    seg_rate = seg_claim / np.maximum(seg_seen, 1e-9)
    return (claim + kappa * seg_rate[sid]) / (den + kappa)


def _split_members(P, members, ncut_max, min_side):
    """Test one linked cluster for two objects glued together; return two member sets or None.

    WHY A CLUSTER NEEDS THIS AT ALL.  link_masks agglomerates masks by 3D overlap WITH TRANSITIVITY:
    A-B and B-C make A, B and C one object.  That is what closes a chain of viewpoints around a
    single chair, and it is also what welds the chair to the table it is pushed under, because some
    mask somewhere overlaps both.  The measured consequence is that our proposals are supersets --
    median purity 0.156 at IoU 0.25 and 0.313 at 0.50, with OVER + BOTH accounting for 54% and 63%
    of misses respectively, on BOTH front ends, so it is the linker and not the masks.

    THE TEST.  Masks of one object overlap each other; masks of two welded objects form two groups
    with little overlap between them.  So build the containment affinity between the cluster's own
    member masks and look for a cheap two-way cut: the Fiedler vector of the normalised Laplacian
    gives the bipartition, and the normalised cut prices it.  A genuine single object has no cheap
    cut and the test declines to split it, which is the property that matters -- this must be far
    more willing to leave a cluster alone than to divide it.

    This is a SPLIT TEST inside one cluster, over its own masks.  It is not the rejected v8 co-claim
    superpoint graph, which built affinity edges BETWEEN SUPERPOINTS across the whole scene and
    clustered them into objects.  Nothing here groups anything, and the graph never leaves a cluster
    the linker already formed.
    """
    n = len(members)
    if n < 2 * min_side:
        return None
    Pm = P[:, members]
    ov = (Pm.T @ Pm).toarray().astype(np.float64)
    sz = np.diag(ov).copy()
    np.fill_diagonal(ov, 0.0)
    A = ov / (np.minimum.outer(sz, sz) + 1e-9)      # containment affinity in [0, 1]
    deg = A.sum(1)
    if (deg <= 1e-9).any():                          # an isolated mask: the cut is trivial, skip
        return None
    inv = 1.0 / np.sqrt(deg)
    L = np.eye(n) - (A * inv[:, None]) * inv[None, :]
    try:
        _w, V = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return None
    fiedler = V[:, 1] * inv
    side = fiedler > 0
    a, b = np.flatnonzero(side), np.flatnonzero(~side)
    if len(a) < min_side or len(b) < min_side:
        return None
    cut = float(A[np.ix_(a, b)].sum())
    va, vb = float(deg[a].sum()), float(deg[b].sum())
    if va <= 0 or vb <= 0:
        return None
    ncut = cut * (1.0 / va + 1.0 / vb)
    if ncut > ncut_max:
        return None
    return members[a], members[b]


def link_masks(d, tau_same=0.45, tau_cross=0.75, veto=0.05, max_ext=3.0, use_veto=True,
               carve=0.0, greedy=False, return_members=False, claim_mode="frame",
               veto_mode="mean", conflict_frac=0.5, return_hierarchy=False,
               spp=None, kappa=0.0, split_ncut=0.0, split_min_side=2, split_mode="replace"):
    """Agglomerate masks into objects by 3D overlap, semantically primed, view-consensus vetoed."""
    P = d["P"].tocsc(); nM = d["nM"]
    if nM == 0: return []
    V = d["V"]; conc = d["mask_concept"].astype(np.int32); mf = d["mask_frame"]
    Vis = d["Vis"]
    # view-consensus rate (MC's quantity) kept as a veto signal
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
    # cluster state: G[m] = cluster id
    G = np.arange(nM)
    hierarchy = {tuple((mask,)) for mask in range(nM)} if return_hierarchy else None
    for _ in range(30):
        cl_ids = np.unique(G); nC = len(cl_ids)
        if nC <= 1: break
        remap = {c: i for i, c in enumerate(cl_ids)}
        idx = np.array([remap[g] for g in G])
        Gm = sp.csr_matrix((np.ones(nM, np.float32), (np.arange(nM), idx)), shape=(nM, nC))
        Pc = (P @ Gm).tocsc(); Pc.data[:] = 1.0                    # cluster x vertex incidence
        sz = np.asarray(Pc.sum(0)).ravel() + 1e-9
        ov = (Pc.T @ Pc).toarray()
        cont = ov / (np.minimum.outer(sz, sz) + 1e-9)              # symmetric containment
        np.fill_diagonal(cont, 0.0)
        # semantic prior: dominant concept per cluster
        cconc = np.zeros(nC, np.int32)
        for c in range(nC):
            mem = conc[idx == c]
            cconc[c] = np.bincount(mem).argmax() if len(mem) else -1
        same = cconc[:, None] == cconc[None, :]
        thr = np.where(same, tau_same, tau_cross)
        cand = cont >= thr
        if use_veto:                                               # separate-object evidence
            Gd = Gm.toarray()                                      # nM x nC (0/1)
            if veto_mode == "mean":
                ob = Gd.T @ obs @ Gd                               # co-observation mass per pair
                sup_c = Gd.T @ (rate0 * obs) @ Gd                  # consensus-weighted support
                rate = sup_c / (ob + 1e-7)                         # mean view-consensus per pair
                cand &= ~((ob >= 3) & (rate < veto))
            elif veto_mode == "conflict-fraction":
                # Mean aggregation can erase a real cannot-link: once a broad articulation mask joins
                # object A, its positive edge to B can dilute the A-vs-B separation evidence and let a
                # transitive merge weld both objects.  Preserve that evidence by measuring the share of
                # independently co-observed mask pairs that explicitly vote for separation.
                eligible_c = Gd.T @ eligible0 @ Gd
                conflict_c = Gd.T @ conflict0 @ Gd
                fraction = conflict_c / (eligible_c + 1e-7)
                cand &= ~((eligible_c >= 1) & (fraction >= conflict_frac))
            else:
                raise ValueError(f"unknown veto mode: {veto_mode}")
        if not cand.any(): break
        # mutual nearest neighbour agglomeration (stable; avoids single-linkage chaining)
        score = np.where(cand, cont, -1.0)
        best = score.argmax(1); merged = False
        newG = G.copy()
        done = np.zeros(nC, bool)
        for a in range(nC):
            b = best[a]
            if score[a, b] < 0 or done[a] or done[b] or a == b: continue
            if best[b] != a: continue                              # require mutual
            gg = V[np.unique(np.concatenate([Pc[:, a].indices, Pc[:, b].indices]))]
            if len(gg) and (gg.max(0) - gg.min(0)).max() > max_ext: continue   # size guard
            newG[idx == b] = cl_ids[a]                             # canonical id = cluster a's
            done[a] = done[b] = True; merged = True
        if not merged and greedy:
            # In DENSE clutter mutual-NN stalls: a cluster's best partner often prefers a third
            # cluster, so no mutual pair exists and agglomeration halts with the scene still
            # shattered (this was the scene0030 failure: 556 boxes for 77 objects). Fall back to
            # greedy highest-overlap-first pairing, which cannot stall, using the same thresholds.
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
    if split_ncut > 0:
        # test each cluster for two welded objects; a cluster that declines the test is untouched
        expanded = []
        for members in member_sets:
            parts = _split_members(P, np.asarray(members), split_ncut, split_min_side)
            if parts is None:
                expanded.append(members)
            elif split_mode == "hedge":
                # keep the union too: the closed form buys one extra true positive for 3.7 charged
                # false positives at IoU 0.25, so hedging a split is favourably priced if it is ever
                # right.  'replace' is the honest partition-form alternative; TRAIN picks between them.
                expanded.extend([members, parts[0], parts[1]])
            else:
                expanded.extend([parts[0], parts[1]])
        member_sets = expanded
    for members in member_sets:
        vids = np.unique(P[:, members].indices)
        nviews = len(np.unique(mf[members]))                       # distinct frames supporting it
        if carve > 0 and len(vids) >= 40:
            # INTRA-CLUSTER CONSENSUS CARVE (our Component-1 mechanism, applied to the linked cluster):
            # a vertex survives only if a sufficient fraction of the frames in which it was VISIBLE to
            # this cluster's own masks actually claimed it. Removes the leak a pure union retains.
            ratio = _consensus_ratio(P, Vis_c, vids, members, mf, claim_mode,
                                     spp=spp, kappa=kappa)
            keep = ratio >= carve
            mcons = float(ratio[keep].mean()) if keep.sum() else 0.0
            if keep.sum() >= 40:
                vids = vids[keep]
        else:
            mcons = 0.5
        out.append((vids, len(members), nviews, mcons) + ((members,) if return_members else ()))
    return out


def make_preds(d, tau_same, tau_cross, veto, use_veto=True, min_pts=40,
               min_masks=1, nms=0.5, rank="views", carve=0.0, contain_nms=1.0,
               soft=0.0, support_w=1.0, greedy=False, claim_mode="frame",
               veto_mode="mean", conflict_frac=0.5):
    """rank: 'size' = point count (naive); 'views' = MULTI-VIEW SUPPORT (calibrated reliability --
    an object confirmed across many independent frames is real; a single-view blob is noise)."""
    V = d["V"]; clouds = []
    for vids, nmask, nviews, mcons in link_masks(
            d, tau_same, tau_cross, veto, use_veto=use_veto, carve=carve, greedy=greedy,
            claim_mode=claim_mode, veto_mode=veto_mode, conflict_frac=conflict_frac):
        if len(vids) < min_pts or nmask < min_masks: continue
        g, lo, hi = best_box(V[vids])
        if len(g) < 20: continue
        # ---- calibrated reliability features ----
        ext = hi - lo; vol = float(np.prod(np.maximum(ext, 1e-6)))
        survive = len(g) / max(len(vids), 1)              # fraction surviving geometric cleaning
        dens = len(g) / max(vol, 1e-6)                    # point density (noise clusters are sparse)
        mx = float(ext.max())
        plaus = float(np.exp(-max(0.0, mx - 2.0)) * (1.0 if mx > 0.10 else 0.2))  # object-scale prior
        views = float(nviews) * (1.0 - np.exp(-len(vids) / 200.0))
        if   rank == "size":     conf = float(len(vids))
        elif rank == "views":    conf = views
        elif rank == "survive":  conf = views * survive
        elif rank == "plaus":    conf = views * plaus
        elif rank == "dens":     conf = views * np.log1p(dens)
        elif rank == "cons":     conf = views * max(mcons, 1e-3)
        else:                    conf = views * survive * plaus * max(mcons, 1e-3)   # "full"
        clouds.append((lo, hi, conf))
    clouds.sort(key=lambda x: -x[2])
    kept = []
    for lo, hi, n in clouds:
        if any(aiou((lo, hi), (k[0], k[1])) > nms for k in kept): continue
        # CONTAINMENT suppression: a fragment box sitting inside an already-accepted object is a
        # duplicate, not a new object -- IoU-NMS misses it (small-in-large has low IoU).
        if contain_nms < 1.0:
            vol = float(np.prod(np.maximum(hi - lo, 1e-6)))
            dup = False
            for klo, khi, _ in kept:
                ilo = np.maximum(lo, klo); ihi = np.minimum(hi, khi)
                inter = float(np.prod(np.clip(ihi - ilo, 0, None)))
                if inter / vol > contain_nms: dup = True; break
            if dup: continue
        kept.append((lo, hi, n))

    # ---- SUPPORT-CONTACT evidence (our thesis; MC cannot use it): a real object RESTS on the floor
    # or on another object's top surface. A cluster floating with no supporter is likely a fragment.
    if support_w > 0 and kept:
        zs = np.array([lo[2] for lo, _, _ in kept])
        floor = float(np.percentile(zs, 5))
        tops = [(float(hi[2]), lo[:2], hi[:2]) for lo, hi, _ in kept]
        resc = []
        for lo, hi, cf in kept:
            base = float(lo[2]); ok = abs(base - floor) < 0.15
            if not ok:
                for tz, tlo, thi in tops:                     # resting on some object's top?
                    if abs(base - tz) < 0.15 and np.all(lo[:2] <= thi + 0.1) and np.all(hi[:2] >= tlo - 0.1):
                        ok = True; break
            resc.append((lo, hi, cf * (1.0 if ok else support_w)))
        kept = resc

    # ---- SOFT duplicate demotion: a box largely inside a higher-ranked one is probably a fragment,
    # but deleting it costs recall (hard containment-NMS above did exactly that). Instead DECAY its
    # confidence so it sinks below the true objects on the PR curve -- recall preserved, AP improved.
    if soft > 0 and kept:
        kept.sort(key=lambda x: -x[2]); out = []
        for i, (lo, hi, cf) in enumerate(kept):
            vol = float(np.prod(np.maximum(hi - lo, 1e-6))); mx = 0.0
            for klo, khi, _ in out:
                ilo = np.maximum(lo, klo); ihi = np.minimum(hi, khi)
                inter = float(np.prod(np.clip(ihi - ilo, 0, None)))
                mx = max(mx, inter / vol)
            out.append((lo, hi, cf * float(np.exp(-soft * mx * mx))))
        kept = sorted(out, key=lambda x: -x[2])
    return kept


def metrics(pbs, gbs):
    best=[]; npred=0; ex=[]
    for pl,gl in zip(pbs,gbs):
        boxes=[((lo+hi)/2,hi-lo) for lo,hi,_ in pl]; npred+=len(boxes)
        for gc,gs in gl: best.append(max((iou(c,s,gc,gs) for c,s in boxes),default=0.))
        m=[]
        for c,s in boxes:
            bj,bv=-1,.25
            for j,(gc,gs) in enumerate(gl):
                v=iou(c,s,gc,gs)
                if v>=bv: bv,bj=v,j
            if bj>=0: m.append((c,s,gl[bj]))
        if len(m)>=2:
            op=[max([ofrac(m[i][0],m[i][1],m[j][0],m[j][1]) for j in range(len(m)) if j!=i]+[0]) for i in range(len(m))]
            og=[max([ofrac(m[i][2][0],m[i][2][1],m[j][2][0],m[j][2][1]) for j in range(len(m)) if j!=i]+[0]) for i in range(len(m))]
            ex+=[max(0.,op[i]-og[i]) for i in range(len(m))]
    def ap(thr):
        E=[(n,si,(lo+hi)/2,hi-lo) for si,pl in enumerate(pbs) for lo,hi,n in pl]; E.sort(key=lambda x:-x[0])
        mt=[np.zeros(len(g),bool) for g in gbs]; ng=sum(len(g) for g in gbs)
        tp=np.zeros(len(E)); fp=np.zeros(len(E))
        for i,(cf,si,c,s) in enumerate(E):
            bj,bv=-1,thr
            for j,g in enumerate(gbs[si]):
                if mt[si][j]: continue
                v=iou(c,s,g[0],g[1])
                if v>=bv: bv,bj=v,j
            if bj>=0: mt[si][bj]=True; tp[i]=1
            else: fp[i]=1
        ctp,cfp=np.cumsum(tp),np.cumsum(fp); rec=ctp/max(ng,1); pr=ctp/np.maximum(ctp+cfp,1)
        mr=np.concatenate([[0],rec,[1]]); mp=np.concatenate([[0],pr,[0]])
        for i in range(len(mp)-1,0,-1): mp[i-1]=max(mp[i-1],mp[i])
        ii=np.where(mr[1:]!=mr[:-1])[0]; return float(np.sum((mr[ii+1]-mr[ii])*mp[ii+1])*100)
    best=np.array(best)
    return dict(npred=npred,miou=best.mean(),r25=(best>=.25).mean(),r50=(best>=.5).mean(),
                ap25=ap(.25),ap50=ap(.5),interp=float(np.mean(ex))*100 if ex else 0.)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tau",type=float,default=0.45); ap.add_argument("--cross",type=float,default=0.75)
    ap.add_argument("--veto",type=float,default=0.05); ap.add_argument("--no-veto",action="store_true")
    ap.add_argument("--sweep",action="store_true"); ap.add_argument("--write",default=None)
    ap.add_argument("--min-masks",type=int,default=1); ap.add_argument("--nms",type=float,default=0.5)
    ap.add_argument("--rank",default="views"); ap.add_argument("--carve",type=float,default=0.20); ap.add_argument("--contain-nms",type=float,default=1.0)
    ap.add_argument("--soft",type=float,default=0.0); ap.add_argument("--support-w",type=float,default=1.0); ap.add_argument("--greedy",action="store_true")
    args=ap.parse_args()
    data={sc:pickle.load(open(f"{V5}/{sc}.pkl","rb")) for sc in SCENES if glob.glob(f"{V5}/{sc}.pkl")}
    gts={}
    for sc in data:
        inst,_=load_gt_instances(f"{SCAN}/{sc}",sc)
        gts[sc]=[(np.asarray(g["center"],float),np.asarray(g["size"],float)) for g in inst]
    ng=sum(len(v) for v in gts.values())
    print(f"  === v6 semantic-geometric mask linking, {len(data)} scenes, {ng} GT ===")
    print(f"  {'method':24s} {'#pred':>6s} {'mIoU':>6s} {'R@.25':>6s} {'R@.5':>6s} {'AP@25':>6s} {'AP@50':>6s} {'interp':>7s}")
    print(f"  {'MaskClustering (SOTA)':24s} {511:6d} {0.579:6.3f} {0.867:6.3f} {0.622:6.3f} {67.9:6.1f} {40.0:6.1f} {2.2:6.1f}%")
    # cfg = (name, tau_same, tau_cross, veto, use_veto, min_masks, nms, rank)
    cfgs=[(f"v6 carve{args.carve} s{args.soft}",args.tau,args.cross,args.veto,not args.no_veto,
           args.min_masks,args.nms,args.rank,args.carve,args.contain_nms,args.soft,args.support_w,args.greedy)]
    if args.sweep:
        cfgs=[("v6 mutual (current)",0.60,0.85,0.05,True,1,0.5,"full",0.20,1.0,2.0,1.0,False),
              ("v6 +greedy",0.60,0.85,0.05,True,1,0.5,"full",0.20,1.0,2.0,1.0,True),
              ("v6 +greedy carve.30",0.60,0.85,0.05,True,1,0.5,"full",0.30,1.0,2.0,1.0,True),
              ("v6 +greedy tau.70",0.70,0.90,0.05,True,1,0.5,"full",0.20,1.0,2.0,1.0,True),
              ("v6 +greedy tau.50",0.50,0.80,0.05,True,1,0.5,"full",0.20,1.0,2.0,1.0,True),
              ("v6 +greedy t.70 c.30",0.70,0.90,0.05,True,1,0.5,"full",0.30,1.0,2.0,1.0,True)]
    for name,ts,tc,vt,uv,mm,nm,rk,cv,cn,sf,sw,gd in cfgs:
        pbs=[make_preds(data[sc],ts,tc,vt,uv,min_masks=mm,nms=nm,rank=rk,carve=cv,contain_nms=cn,soft=sf,support_w=sw,greedy=gd) for sc in data]
        gbs=[gts[sc] for sc in data]; r=metrics(pbs,gbs)
        print(f"  {name:24s} {r['npred']:6d} {r['miou']:6.3f} {r['r25']:6.3f} {r['r50']:6.3f} {r['ap25']:6.1f} {r['ap50']:6.1f} {r['interp']:6.1f}%")
    if args.write:
        out={}
        for sc in data:
            kept=make_preds(data[sc],args.tau,args.cross,args.veto,not args.no_veto,
                            min_masks=args.min_masks,nms=args.nms,rank=args.rank,carve=args.carve,contain_nms=args.contain_nms,soft=args.soft,support_w=args.support_w,greedy=args.greedy)
            ns=[n for _,_,n in kept] or [1]; mx=max(ns)
            out[sc]=[["obj",*(((lo+hi)/2).tolist()),*((hi-lo).tolist()),float(np.log1p(n)/np.log1p(mx))]
                     for lo,hi,n in kept]
        json.dump(out,open(args.write,"w")); print(f"[wrote] {args.write}")


if __name__=="__main__":
    main()
