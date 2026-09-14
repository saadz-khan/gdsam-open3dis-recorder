"""Sparse implementation of the frozen SceneInformation objective and solver.

Only contingency storage and row aggregation differ. All objective terms,
candidate moves, mass constraints and optimization decisions are inherited.
"""
import numpy as np
from scipy import sparse
from scipy.special import xlogy
from c1_visibility_information import SceneInformation

class SparseSceneInformation(SceneInformation):
    def __init__(self,evidence,anchored=True):
        self.anchored=bool(anchored);self._term_cache={}
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
            block=(e.S.T@p.multiply(inv[:,None])).tocsr()
            bg=np.maximum(self.T[:,frame]-np.asarray(block.sum(1)).ravel(),0)
            foreground.append(self.T[:,frame]-bg)
            block=sparse.hstack([block,sparse.csr_matrix(bg[:,None])],format='csr')
            blocks.append(block);colframes.extend([frame]*block.shape[1])
            np.testing.assert_allclose(np.asarray(block.sum(1)).ravel(),self.T[:,frame],atol=1e-8)
        self.FG=np.stack(foreground,axis=1)
        self.C=sparse.hstack(blocks,format='csr');self.cf=np.asarray(colframes,np.int32)
        colmass=np.asarray(self.C.sum(0)).ravel()
        self.constant=np.bincount(self.cf,weights=xlogy(colmass,colmass),minlength=e.nF)/np.maximum(self.N,1)

    def term(self,mask,background=False):
        if not np.any(mask):return np.zeros(self.e.nF)
        key=(self.anchored,background,np.packbits(mask).tobytes())
        if key in self._term_cache:return self._term_cache[key]
        cells=np.asarray(self.C[mask].sum(0)).ravel();rows=self.T[mask].sum(0)
        entropy_cells=np.bincount(self.cf,weights=xlogy(cells,cells),minlength=self.e.nF)
        numerator=xlogy(rows,rows)-2*entropy_cells
        if self.anchored:
            fg=self.FG[mask].sum(0)
            numerator+=np.log(2.)*(fg if background else rows-fg)
        answer=numerator/np.maximum(self.N,1);self._term_cache[key]=answer
        return answer
