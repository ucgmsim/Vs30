import numpy as np
from sklearn.cluster import DBSCAN

ID_NODATA = 255

def cluster(sites, letter, min_group=5, eps=15000, plots=False):
    """
    plots: make figures, not expected to be used very often
    """

    features = np.column_stack((sites.easting.values, sites.northing.values))
    # default not a member of any cluster (-1)
    sites[f"{letter}cluster"] = -1
    mids = sites[f"{letter}id"].values
    ids = np.array(sorted(set(mids)))
    ids = ids[ids != ID_NODATA].astype(np.int)

    for i in ids:
        subset = features[mids == i]
        if subset.shape[0] < min_group:
            # can't form any groups
            continue

        dbscan = DBSCAN(eps=eps, min_samples=min_group, n_jobs=-1)
        dbscan.fit(subset)

        # save labels
        sites.loc[mids == i, f"{letter}cluster"] = dbscan.labels_

    return sites
