#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

PLOTS = False
VS_FILE = "data/vspr.csv"

if PLOTS:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from sklearn.metrics import silhouette_score
    import shapefile

# x y longitude latitude
vspr = pd.read_csv(VS_FILE)
# not a member of any cluster (-1)
vspr["cluster_aak"] = -1
vspr["cluster_yca"] = -1
features = np.array([vspr.x, vspr.y]).T

for model in ["aak", "yca"]:
    gids = vspr[f"gid_{model}"].values
    ids = np.array(sorted(set(gids)))
    ids = ids[np.invert(np.isnan(ids))].astype(np.int)

    for i in ids:
        i_idxs = np.where(gids == i)[0]
        scaled_subset = features[i_idxs]
        if scaled_subset.shape[0] < 4:
            # all in different subgroup
            continue

        dbscan = DBSCAN(eps=15000, min_samples=5, n_jobs=-1)
        dbscan.fit(scaled_subset)

        # save labels
        vspr.loc[i_idxs, f"cluster_{model}"] = dbscan.labels_
        if not PLOTS:
            continue

        # Compute the silhouette score
        if len(set(dbscan.labels_)) > 1:
            dbscan_silhouette = silhouette_score(scaled_subset, dbscan.labels_).round(2)
        else:
            dbscan_silhouette = "na"

        # Plot the data and cluster silhouette comparison
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(8, 6), sharex=False, sharey=False
        )
        fig.suptitle(f"Clustering Measured Sites", fontsize=16)
        s = shapefile.Reader("/run/media/vap30/Hathor/work/plotting_data/Paths/CFM_v0_4_Review/Shapefiles/Coastline_QMap.shp")
        r = s.shapeRecords()
        for p in r:
            ax2.plot(*list(zip(*p.shape.points)), color="black", linewidth=0.8)
        hv = []
        hk = {}
        for v in dbscan.labels_:
            if v == -1:
                hv.append(1)
            else:
                if v in hk:
                    hk[v] += 1
                else:
                    hk[v] = 1
        for v in hk.values():
            hv.append(v)
        ax1.hist(hv, bins=50)
        for label in set(dbscan.labels_):
            idx = np.where(dbscan.labels_ == label)[0]
            print(label, idx)
            ax2.scatter(scaled_subset[idx, 0], scaled_subset[idx, 1], s=20)
        ax2.set_title(
            f"Silhouette score: {dbscan_silhouette}", fontdict={"fontsize": 12}
        )
        hk[-1] = 1
        hvw = 1/np.array([hk[v] for v in dbscan.labels_])
        plt.savefig("cluster-" + model + "-" + str(i) + ".png")
        plt.close()

vspr.to_csv(VS_FILE, index=False, na_rep="NA")
