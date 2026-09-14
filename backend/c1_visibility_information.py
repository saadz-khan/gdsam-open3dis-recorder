"""Joint scene partition inference from visibility-censored information loss.

Each view supplies a fractional mask partition (including explicit background).
The inferred 3D partition is projected by measured depth visibility. Minimize
mean variation of information between the two partitions, rather than scoring
objects independently. Candidate moves can birth, split or replace objects;
every accepted move strictly decreases the declared scene objective.

Variation of information and fractional-mask evidence are established tools.
Their use here for transactional 3D hypothesis selection is an experimental
contribution whose novelty and empirical advantage remain to be established.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.special import xlogy
from c1_crossview_attractors import Attractors


class SceneInformation:
    def __init__(self,evidence,anchored=True):
        self.anchored=bool(anchored)
        self._term_cache={}
        e=evidence;self.e=e;self.total=e.total.astype(np.float64)
        self.T=e.T.toarray().astype(np.float64)
        self.N=self.T.sum(0);self.valid=self.N>0
        blocks=[];colframes=[];foreground=[]
        for frame,ids in enumerate(e.frames):
            p=e.P[:,ids].tocsr().astype(np.float64)
            vis=e.Vis[:,frame].toarray().ravel()
            if p.shape[1] and np.any(np.asarray(p[vis==0].sum(1)).ravel()>0):
                raise ValueError('Mask claims outside measured visibility')
            counts=np.asarray(p.sum(1)).ravel()
            inv=np.divide(1.,counts,out=np.zeros_like(counts),where=counts>0)
            q=p.multiply(inv[:,None])
            block=(e.S.T@q).toarray()
            bg=np.maximum(self.T[:,frame]-block.sum(1),0)
            foreground.append(self.T[:,frame]-bg)
            block=np.column_stack([block,bg]);blocks.append(block)
            colframes.extend([frame]*block.shape[1])
            np.testing.assert_allclose(block.sum(1),self.T[:,frame],atol=1e-8)
        self.FG=np.stack(foreground,axis=1)
        self.C=np.concatenate(blocks,axis=1)
        self.cf=np.asarray(colframes,np.int32)
        colmass=self.C.sum(0)
        self.constant=np.bincount(self.cf,weights=xlogy(colmass,colmass),minlength=e.nF)/np.maximum(self.N,1)

    def term(self,mask,background=False):
        if not np.any(mask):return np.zeros(self.e.nF)
        key=(self.anchored,background,np.packbits(mask).tobytes())
        if key in self._term_cache:return self._term_cache[key]
        cells=self.C[mask].sum(0);rows=self.T[mask].sum(0)
        entropy_cells=np.bincount(self.cf,weights=xlogy(cells,cells),minlength=self.e.nF)
        numerator=xlogy(rows,rows)-2*entropy_cells
        if self.anchored:
            fg=self.FG[mask].sum(0)
            # Pointwise Jensen-Shannon divergence between hard foreground bits:
            # zero for agreement and ln(2) for disagreement, in the same nats as VI.
            numerator += np.log(2.)*(fg if background else rows-fg)
        answer=numerator/np.maximum(self.N,1)
        self._term_cache[key]=answer
        return answer

    def value(self,owners):
        return self.constant+sum((self.term(owners==k,background=k==0) for k in np.unique(owners)),start=np.zeros(self.e.nF))

    def propose(self,owners,birthmass,mask,new_id,terms):
        changed=np.unique(owners[mask])
        # Relabeling a complete existing object changes no scene partition.
        if len(changed)==1 and changed[0]!=0 and np.array_equal(owners==changed[0],mask):return None
        nxt=owners.copy();nxt[mask]=new_id
        affected=set(map(int,changed));affected.add(0)
        for k in changed:
            if k==0:continue
            remaining=nxt==k;mass=float(self.total[remaining].sum())
            if mass<100 or mass<.65*birthmass[int(k)]:nxt[remaining]=0
        newterms={k:self.term(nxt==k,background=k==0) for k in affected}
        newterms[new_id]=self.term(mask)
        delta=newterms[new_id].copy()
        for k in affected:delta+=newterms[k]-terms.get(k,0)
        return nxt,newterms,delta

    def optimize(self,initial,modes,mode='mean',max_steps=32):
        if mode not in {'mean','both'}:raise ValueError(mode)
        owners=np.asarray(initial,np.int32).copy()
        mass={int(k):float(self.total[owners==k].sum()) for k in np.unique(owners) if k>0}
        terms={int(k):self.term(owners==k,background=k==0) for k in np.unique(owners)}
        energies=[self.value(owners)];transactions=[]
        for step in range(max_steps):
            new_id=int(owners.max())+1
            best=None;best_gain=-1e-10
            for index,mask in enumerate(modes):
                candidate=self.propose(owners,mass,mask,new_id,terms)
                if candidate is None:continue
                nxt,newterms,delta=candidate
                gain=float(delta[self.valid].mean())
                if gain>=best_gain:continue
                if mode=='both':
                    valid=True
                    for parity in range(2):
                        f=self.valid & (np.arange(self.e.nF)%2==parity)
                        if not f.any() or float(delta[f].mean())>=-1e-10:valid=False;break
                    if not valid:continue
                best=(index,nxt,newterms,delta);best_gain=gain
            if best is None:break
            index,nxt,newterms,delta=best
            before=self.value(owners);after=self.value(nxt)
            np.testing.assert_allclose(after-before,delta,atol=1e-10)
            assert float((after-before)[self.valid].mean())<-1e-10
            transactions.append({'hypothesis':index,'mean_delta':best_gain,
                'fold_delta':[float(delta[self.valid & (np.arange(self.e.nF)%2==p)].mean()) for p in range(2)],
                'changed_superpoints':int((nxt!=owners).sum())})
            owners=nxt;terms.update(newterms);mass[new_id]=float(self.total[modes[index]].sum())
            for k in list(terms):
                if not np.any(owners==k):terms.pop(k,None);mass.pop(k,None)
            energies.append(after)
        return owners,{'transactions':transactions,'initial_vi':float(energies[0][self.valid].mean()),
            'final_vi':float(energies[-1][self.valid].mean()),'accepted':len(transactions),
            'iteration_limit':len(transactions)==max_steps,'objective_monotone':True}


def refine(item,incumbent):
    e=Attractors(item['recording'],item['superpoints']);allframes=np.arange(e.nF)
    bank,_=e.evolve(e.seeds(allframes),allframes,one_step=True)
    _,_,scores=e.correspond(bank,allframes)
    bank=bank[(scores>=.7)&((bank@e.total)>=100)]
    owners=np.zeros(e.nS,np.int32)
    for k,p in enumerate(incumbent['preds'],1):
        sp=np.unique(e.spp[p['v']]);assert np.all(owners[sp]==0),'Incumbent must be disjoint'
        assert int(e.total[sp].sum())==len(p['v']),'Incumbent must contain whole superpoints'
        owners[sp]=k
    energy=SceneInformation(e);results={};diagnostics={'hypotheses':len(bank)}
    for name,mode,anchor in [('vi_unanchored','mean',False),('vi_mean','mean',True),('vi_both','both',True)]:
        energy.anchored=anchor
        final,diag=energy.optimize(owners,bank,mode)
        masks=np.array([final==k for k in np.unique(final) if k>0],bool).reshape(-1,e.nS)
        _,_,score=e.correspond(masks,allframes)
        results[name]={'nV':e.nV,'preds':[{'v':np.flatnonzero(row[e.spp]).astype(np.int32),'conf':float(sc)} for row,sc in zip(masks,score)]}
        diagnostics[name]=diag
    return results,diagnostics
