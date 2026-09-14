"""Training-free proposal birth from independently reconstructed view subsets.

Raw-mask seeds evolve by alternating discrete mask correspondence and superpoint
occupancy. No irreversible agglomeration is used. A fixed point is accepted only
when another update leaves it identical; cycles and iteration limits are reported
and rejected. Crossfit proposals must be independently rediscovered in both folds.
All thresholds are inherited from deployed reconstruction or fixed before GT.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse


def unique_rows(b):
    b = np.asarray(b, bool)
    if not len(b): return b
    packed = np.packbits(b, axis=1, bitorder='little')
    _, ids = np.unique(packed, axis=0, return_index=True)
    return b[ids]


class Attractors:
    def __init__(self, recording, superpoints):
        self.spp = np.asarray(superpoints, np.int32)
        self.nV = len(self.spp)
        if self.nV == 0 or self.spp.min() < 0:
            raise ValueError('Expected nonempty nonnegative superpoints')
        self.nS = int(self.spp.max()) + 1
        self.P = recording['P'][:self.nV].tocsr().astype(np.float32)
        self.Vis = recording['Vis'][:self.nV].tocsr().astype(np.float32)
        self.mf = np.asarray(recording['mask_frame'], np.int32)
        self.nF = self.Vis.shape[1]
        if len(self.mf) != self.P.shape[1] or np.any(self.mf < 0) or np.any(self.mf >= self.nF):
            raise ValueError('Invalid mask-frame alignment')
        if not np.all(self.P.data == 1) or not np.all(self.Vis.data == 1):
            raise ValueError('Binary recording required')
        self.S = sparse.csr_matrix((np.ones(self.nV, np.float32),
                         (np.arange(self.nV), self.spp)), shape=(self.nV,self.nS))
        self.C = (self.S.T @ self.P).tocsr()
        self.T = (self.S.T @ self.Vis).tocsr()
        self.mass = np.asarray(self.P.sum(0)).ravel()
        self.total = np.bincount(self.spp, minlength=self.nS).astype(np.float32)
        # Canonicalize mask ties by geometry, so recording-column order cannot
        # decide which identical-score witness wins. Exact duplicates are removed.
        self.frames = []
        pc = self.P.tocsc()
        for f in range(self.nF):
            ids = np.flatnonzero(self.mf == f)
            keys = [(tuple(pc.indices[pc.indptr[i]:pc.indptr[i+1]]), int(i)) for i in ids]
            keep, previous = [], None
            for key, i in sorted(keys):
                if key != previous: keep.append(i)
                previous = key
            self.frames.append(np.asarray(keep, np.int32))

    def seeds(self, frames):
        ids = np.concatenate([self.frames[f] for f in frames]) if len(frames) else np.empty(0,np.int32)
        if not len(ids): return np.zeros((0,self.nS),bool)
        c = self.C[:, ids].toarray().T
        t = self.T[:, self.mf[ids]].toarray().T
        b = (c >= .5 * t) & (c > 0)
        return unique_rows(b[(b @ self.total) >= 40])

    def correspond(self, b, frames, top_k=20, floor=.5):
        n = len(b)
        if not n: return np.zeros((0,self.P.shape[1]),np.float32), np.zeros((0,self.nF),np.float32), np.zeros(0)
        sb = sparse.csr_matrix(b, dtype=np.float32)
        inter = (sb @ self.C).toarray()
        vis = (sb @ self.T).toarray()
        eligible = np.zeros((n,self.nF),bool)
        frames = np.asarray(frames,np.int32)
        order = np.argsort(-vis[:,frames],axis=1,kind='stable')[:,:top_k]
        np.put_along_axis(eligible,frames[order],True,axis=1)
        eligible &= vis > 0
        q = np.zeros((n,self.P.shape[1]),np.float32)
        qf = np.zeros((n,self.nF),np.float32)
        score = np.zeros((n,self.nF),np.float32)
        measured = np.zeros_like(eligible)
        rows = np.arange(n)
        for f in frames:
            ids = self.frames[f]
            if not len(ids): continue
            denom = np.sqrt(vis[:,f,None] * self.mass[ids][None,:])
            agreement = np.divide(inter[:,ids],denom,out=np.zeros_like(denom),where=denom>0)
            best = agreement.argmax(1)
            val = agreement[rows,best]
            good = eligible[:,f] & (val >= floor)
            q[rows[good],ids[best[good]]] = 1
            qf[good,f] = 1
            score[:,f] = val
            measured[:,f] = eligible[:,f]
        # Include observed mismatches (zero agreement) in validation score.
        sc = (score*measured).sum(1)/np.maximum(measured.sum(1),1)
        return q, qf, sc

    def update(self, b, frames):
        q, qf, score = self.correspond(b,frames)
        claim = (sparse.csr_matrix(q) @ self.C.T).toarray()
        seen = (sparse.csr_matrix(qf) @ self.T.T).toarray()
        nxt = (claim >= .3 * seen) & (claim > 0)
        valid = ((nxt @ self.total) >= 40) & (qf.sum(1) >= 2)
        nxt[~valid] = False
        return nxt, score

    def evolve(self, seeds, frames, max_steps=6, one_step=False):
        active = unique_rows(seeds)
        stable, history, diag = [], set(), {'seeds':len(active),'iterations':0,'revisited_states':0,'invalid':0}
        for step in range(1 if one_step else max_steps):
            if not len(active): break
            before = active
            after, _ = self.update(before,frames)
            diag['iterations'] = step+1
            valid = after.any(1)
            diag['invalid'] += int((~valid).sum())
            same = (after == before).all(1) & valid
            if one_step:
                stable.extend(after[valid])
                active = np.zeros((0,self.nS),bool)
                break
            stable.extend(after[same])
            following = unique_rows(after[valid & ~same])
            novel = []
            # Revisited states merge with an already explored trajectory. This
            # detects both duplicate paths and cycles, not a pure cycle count.
            history.update(np.packbits(row).tobytes() for row in before)
            for row in following:
                key = np.packbits(row).tobytes()
                if key in history: diag['revisited_states'] += 1
                else: novel.append(row)
            active = np.asarray(novel,bool).reshape(-1,self.nS)
        diag['iteration_limit'] = len(active)
        result = unique_rows(np.asarray(stable,bool).reshape(-1,self.nS))
        diag['stable'] = len(result)
        return result,diag

    def _packed_size(self):
        return (self.nS+7)//8

    def certified_pairs(self, a, b):
        if not len(a) or not len(b): return np.zeros((0,self.nS),bool),0
        inter = (a.astype(np.float32)*self.total) @ b.T.astype(np.float32)
        union = (a @ self.total)[:,None]+(b @ self.total)[None,:]-inter
        iou = inter/np.maximum(union,1)
        ab, ba = iou.argmax(1),iou.argmax(0)
        # Independent generation, followed by reciprocal pairing and cross-view
        # prediction. Scoring existing all-view proposals on folds is insufficient.
        _,_,ascore = self.correspond(a,np.arange(1,self.nF,2))
        _,_,bscore = self.correspond(b,np.arange(0,self.nF,2))
        ids = [i for i,j in enumerate(ab) if ba[j]==i and iou[i,j]>=.5 and min(ascore[i],bscore[j])>=.7]
        if not ids: return np.zeros((0,self.nS),bool),0
        return unique_rows(a[ids] | b[ab[ids]]),len(ids)

    def generate(self):
        allframes = np.arange(self.nF)
        seed = self.seeds(allframes)
        once,d1 = self.evolve(seed,allframes,one_step=True)
        full,d2 = self.evolve(seed,allframes)
        folds, ds = [], []
        for parity in range(2):
            f = np.arange(parity,self.nF,2)
            m,d = self.evolve(self.seeds(f),f)
            folds.append(m); ds.append(d)
        pairs,n = self.certified_pairs(*folds)
        joint,d3 = self.evolve(pairs,allframes)
        return {'one_step':once,'attractor':full,'crossfit':joint}, {'one_step':d1,'attractor':d2,'folds':ds,'pairs':n,'joint':d3}


def refine(item, original):
    import c1_projection_consistency as pc
    from c1_absorb import build
    e = Attractors(item['recording'],item['superpoints'])
    modes,diag = e.generate()
    baseline = []
    for p in original['raw']['preds']:
        mask = np.zeros(e.nV,bool); mask[p['v']] = True
        baseline.append({'scan_id':item['scene'],'label_id':1,'pred_mask':mask,'conf':p['conf'],'proposal_path':'baseline'})
    results = {}
    for name,b in modes.items():
        _,_,scores = e.correspond(b,np.arange(e.nF))
        additions = [{'scan_id':item['scene'],'label_id':1,'pred_mask':row[e.spp], 'conf':float(sc),'proposal_path':name}
                     for row,sc in zip(b,scores) if sc>=.7]
        diag[name+'_accepted'] = len(additions)
        if not additions:
            results[name] = original['final']
            continue
        pool = baseline+additions
        # The baseline raw confidence is already projection consistency.
        score = np.array([p['conf'] for p in pool])
        selected = pc.rescore_and_select(pool,score,.7,.5,preserve_paths=frozenset({'baseline'}))
        raw = {'nV':e.nV,'preds':[{'v':np.flatnonzero(p['pred_mask']).astype(np.int32),'conf':float(p['conf'])} for p in selected]}
        results[name] = build({item['scene']:raw},[item['scene']],{item['scene']:item['superpoints']},.65,retention='mass')[item['scene']]
    return results,diag
