from __future__ import annotations

from .npydirdataset import NpyDirDataset

#-----------------------------------------------------------------------------

class ClusteredL1Puppi(NpyDirDataset):
    """Pre-clustered, zero-padded L1Puppi jet constituents (CL1P).

    Clustering + padding are done by `scripts/cluster_l1puppi.py`: each .npy file 
    holds dense (n_jets, m) arrays per channel (jets padded/truncated to a fixed 
    m), plus a `real` mask (1 = genuine constituent, 0 = padding). 
    """

    s_CHANNELS = [ "pt", "eta", "phi", "real" ]
    s_LOCATION_ENV = "CLUSTERED_L1PUPPI_LOCATION"
