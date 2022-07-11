import sys
from pathlib import Path
import pandas as pd
import numpy as np


if __name__ == "__main__":
    assert len(sys.argv)==3, f"Usage: {sys.argv[0]} .ll vs30points.csv"

    ll=Path(sys.argv[1])
    vs30points=sys.argv[2]

    ll_df=pd.read_csv(ll,sep=" ",names=["lon","lat","stat_name"],index_col=2)
    vs30_df=pd.read_csv(vs30points)

    vs30_file = ll.with_suffix('.vs30')
    vs30_file = Path(vs30points).parent.resolve()/vs30_file.name

    with open(vs30_file,"w") as f:

        for i in range(len(ll_df)):
            stat=ll_df.iloc[i].name
            station=ll_df.loc[stat]
            vs30point=vs30_df.iloc[i]
            assert np.isclose(station.lon, vs30point.longitude)
            assert np.isclose(station.lat, vs30point.latitude)
            if np.isnan(vs30point.mvn_vs30):
                if np.isnan(vs30point.terrain_mvn_vs30):
                    if np.isnan(vs30point.geology_mvn_vs30):
                        vs30_est=500
                        print(f"{stat} is NaN both Geology and Terain. Default 500 assigned  !500")
                    else:
                        vs30_est=vs30point.geology_mvn_vs30
                        print(f"{stat} only has Geology-based calc. {vs30_est} assigned !G")

                else:
                     vs30_est=500
                     print(f"{stat} only has Terrain-based calc {vs30point.terrain_mvn_vs30}, but default 500 assigned !T500")

            else:
                vs30_est = vs30point.mvn_vs30
             
            f.write(f"{stat} {vs30_est}\n")
    
    print(f"Wrote {vs30_file}")
