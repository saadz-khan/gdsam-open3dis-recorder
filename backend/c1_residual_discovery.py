"""Training-free residual object discovery, validated on two fixed frontends.

Reconstruct hypotheses from raw mask seeds once, then add only recertified
unclaimed geometry. Existing predictions are preserved exactly. This preserves
per-object best-IoU recall; it does not guarantee AP (false births remain possible).
The selected algorithm is the simpler control from the attractor experiments.
"""
import numpy as np
from c1_crossview_attractors import Attractors
from c1_constrained_births import admit


def discover(item, incumbent):
    evidence=Attractors(item['recording'],item['superpoints'])
    frames=np.arange(evidence.nF)
    modes,diagnostics=evidence.evolve(evidence.seeds(frames),frames,one_step=True)
    result,births=admit(evidence,modes,incumbent,crossfit=False)
    diagnostics['births']=births
    return result,diagnostics
