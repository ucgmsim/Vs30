"""
Handles distance based clustering of measured site locations.
Reduces influence of locations with more measured sites than those more sparse.
"""
import numpy as np
from sklearn.cluster import DBSCAN

from vs30.model import ID_NODATA


def cluster(sites, letter, min_group=5, eps=15000, nproc=-1):
    """
    Sort sites into clusters spatially.
    letter: which id? "t"(id) for terrain or "g"(id) for geology
    min_group: the minimum group size
    eps: (metres) how far points are to be considered a different cluster
    nproc: -1 to detect available cores
    """

    features = np.column_stack((sites.easting.values, sites.northing.values))
    # default not a member of any cluster (-1)
    sites[f"{letter}cluster"] = -1
    model_ids = sites[f"{letter}id"].values
    ids = np.array(sorted(set(model_ids)))
    ids = ids[ids != ID_NODATA].astype(np.int)

    for i in ids:
        subset = features[model_ids == i]
        if subset.shape[0] < min_group:
            # can't form any groups
            continue

        dbscan = DBSCAN(eps=eps, min_samples=min_group, n_jobs=nproc)
        dbscan.fit(subset)

        # save labels
        sites.loc[model_ids == i, f"{letter}cluster"] = dbscan.labels_

    return sites
