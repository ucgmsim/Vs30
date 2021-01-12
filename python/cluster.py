#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

PLOTS = True

if PLOTS:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from sklearn.metrics import silhouette_score
    import shapefile

# x y longitude latitude
vspr = pd.read_csv("data/vspr.csv")
ids = np.array(sorted(set(vspr.gid_aak.values)))
ids = ids[np.invert(np.isnan(ids))].astype(np.int)
features = np.array([vspr.x, vspr.y]).T

for i in ids:
    scaled_subset = features[vspr.gid_aak.values == i]
    if scaled_subset.shape[0] < 4:
        # all in different subgroup
        print(str(i) + " too short")
        continue

    dbscan = DBSCAN(eps=15000, min_samples=5, n_jobs=-1)
    dbscan.fit(scaled_subset)

    # save labels back
    #writecsv
    if not PLOTS:
        return

    # Compute the silhouette score
    if len(set(dbscan.labels_)) > 1:
        dbscan_silhouette = silhouette_score(scaled_subset, dbscan.labels_).round (2)
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
        #ax1.plot(*list(zip(*p.shape.points)), color="black", linewidth=0.8)
        ax2.plot(*list(zip(*p.shape.points)), color="black", linewidth=0.8)
    # The k-means plot
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
    #km_colors = [fte_colors[label] for label in kmeans.labels_]
    ax1.hist(hv, bins=50)
    #ax1.scatter(scaled_subset[:, 0], scaled_subset[:, 1], c=km_colors, s=20)
    #ax1.set_title(
    #    f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
    #)
    
    db_colors = [fte_colors[label] for label in dbscan.labels_]#clusterer.labels_]
    ax2.scatter(scaled_subset[:, 0], scaled_subset[:, 1], s=20)
    ax2.set_title(
        f"Silhouette score: {dbscan_silhouette}", fontdict={"fontsize": 12}
    )
    hk[-1] = 1
    hvw = 1/np.array([hk[v] for v in dbscan.labels_])
    plt.savefig("kvh-" + str(i) + ".png")
    plt.close()
    
