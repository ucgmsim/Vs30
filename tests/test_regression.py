import os
import tempfile
from pathlib import Path

import pandas as pd

from vs30 import params, vs30calc

INPUT_SITE_FILE = Path(__file__).parent / "resources"/ "test_sites.csv"
EXPECTED_OUTPUT_DIR = Path(__file__).parent / "expected_location_results"

def test_vs30calc_regression():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_paths = params.PathsParams(
            out=tmpdir,
            overwrite=True,
        )
        p_ll = params.LLFileParams(
            ll_path=str(INPUT_SITE_FILE),
            lon_col_ix=1,
            lat_col_ix=2,
            col_sep=",",
        )

        vs30calc.run_vs30calc(
            p_paths=p_paths,
            p_sites=params.SitesParams(),
            p_grid=params.GridParams(),
            p_ll=p_ll,
            p_geol=params.GeologyParams(),
            p_terr=params.TerrainParams(),
            p_comb=params.CombinationParams(),
            n_procs=os.cpu_count(),
        )

        expected_vs30_df = pd.read_csv(
            EXPECTED_OUTPUT_DIR / "vs30points.csv")
        output_vs30_df = pd.read_csv(
            Path(tmpdir) / "vs30points.csv")
        
        pd.testing.assert_frame_equal(output_vs30_df, expected_vs30_df)


            








