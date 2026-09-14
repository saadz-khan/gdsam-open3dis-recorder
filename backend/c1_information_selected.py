"""Selected joint inference path. Same anchored objective as frozen H5 vi_mean."""
import numpy as np
from c1_crossview_attractors import Attractors
from c1_information_sparse import SparseSceneInformation as SceneInformation


def refine_scene(item, incumbent):
    e=Attractors(item['recording'],item['superpoints'])
    frames=np.arange(e.nF)
    bank,_=e.evolve(e.seeds(frames),frames,one_step=True)
    _,_,scores=e.correspond(bank,frames)
    bank=bank[(scores>=.7)&((bank@e.total)>=100)]
    owners=np.zeros(e.nS,np.int32)
    for k,p in enumerate(incumbent['preds'],1):
        sp=np.unique(e.spp[p['v']])
        if np.any(owners[sp]) or int(e.total[sp].sum())!=len(p['v']):
            raise ValueError('Incumbent must be a disjoint partition of whole superpoints')
        owners[sp]=k
    energy=SceneInformation(e,anchored=True)
    final,diag=energy.optimize(owners,bank,mode='mean')
    masks=np.array([final==k for k in np.unique(final) if k>0],bool).reshape(-1,e.nS)
    _,_,scores=e.correspond(masks,frames)
    result={'nV':e.nV,'preds':[{'v':np.flatnonzero(row[e.spp]).astype(np.int32),'conf':float(sc)}
                            for row,sc in zip(masks,scores)]}
    diag['hypotheses']=len(bank)
    return result,diag
