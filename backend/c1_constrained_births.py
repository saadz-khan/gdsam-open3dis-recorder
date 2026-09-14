"""Object birth by ownership-constrained, independently replicated reconstruction.

The accepted partition defines a feasible set of superpoints. Apply its projector
INSIDE each correspondence/occupancy update, so geometry is reconstructed under
ownership constraints rather than merely clipped after global reconstruction.
Two independent view subsets must discover overlapping stable hypotheses.
"""
import numpy as np
from c1_crossview_attractors import Attractors, unique_rows


class ConstrainedAttractors(Attractors):
    def __init__(self, recording, superpoints, allowed):
        super().__init__(recording,superpoints)
        self.allowed=np.asarray(allowed,bool)
        if self.allowed.shape != (self.nS,):raise ValueError('Invalid feasible set')

    def seeds(self, frames):
        b=super().seeds(frames) & self.allowed
        return unique_rows(b[(b @ self.total)>=40])

    def update(self, b, frames):
        # Clipping only AFTER generation is a separate mandatory ablation.
        nxt,score=super().update(b & self.allowed,frames)
        nxt &= self.allowed
        nxt[(nxt @ self.total)<40]=False
        return nxt,score


def admit(e, modes, original, crossfit=False):
    """Preserve incumbent geometry and recertify every actual residual birth."""
    claimed=np.zeros(e.nS,bool)
    for p in original['preds']:claimed[np.unique(e.spp[p['v']])]=True
    _,_,scores=e.correspond(modes,np.arange(e.nF))
    order=np.argsort(-scores,kind='stable')
    new=[]
    for i in order:
        row=modes[i] & ~claimed
        before=float(modes[i] @ e.total); after=float(row @ e.total)
        if after<100 or after<.65*before:continue
        score=float(e.correspond(row[None,:],np.arange(e.nF))[2][0])
        if score<.7:continue
        if crossfit:
            passed=True
            for parity in range(2):
                _,qf,s=e.correspond(row[None,:],np.arange(parity,e.nF,2))
                if qf.sum()<2 or s[0]<.7:passed=False;break
            if not passed:continue
        vertices=np.flatnonzero(row[e.spp]).astype(np.int32)
        new.append({'v':vertices,'conf':score})
        claimed |= row
    return {'nV':original['nV'],'preds':list(original['preds'])+new},len(new)


def refine(item, original):
    e=Attractors(item['recording'],item['superpoints'])
    modes,base_diag=e.generate()
    allowed=np.ones(e.nS,bool)
    for p in original['final']['preds']:allowed[np.unique(e.spp[p['v']])]=False
    c=ConstrainedAttractors(item['recording'],item['superpoints'],allowed)
    folds=[];ds=[]
    for parity in range(2):
        frames=np.arange(parity,c.nF,2)
        b,d=c.evolve(c.seeds(frames),frames)
        folds.append(b);ds.append(d)
    pair,n=c.certified_pairs(*folds)
    joint,jdiag=c.evolve(pair,np.arange(c.nF))
    results,stats={}, {'unconstrained':base_diag,'constrained_folds':ds,'constrained_pairs':n,'constrained_joint':jdiag}
    for name,bank,certificate in [('protected_one_step',modes['one_step'],False),
                                  ('protected_crossfit',modes['crossfit'],True),
                                  ('constrained_crossfit',joint,True)]:
        results[name],stats[name+'_births']=admit(e,bank,original['final'],certificate)
        # The preservation claim is about explicit masks, not AP or ranked metrics.
        for old,new in zip(original['final']['preds'],results[name]['preds']):
            np.testing.assert_array_equal(old['v'],new['v'])
            assert old['conf']==new['conf']
    return results,stats
