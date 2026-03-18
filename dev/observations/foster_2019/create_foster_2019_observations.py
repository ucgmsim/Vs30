"""
Create the clean combined observation CSVs for the Foster 2019 version of the Vs30 model.

This script was originally used to generate the observation CSV files that ship
with the vs30 package:
  - foster_2019_measured_vs30_independent_observations.csv
  - viktor_inferred_vs30_from_cpt.csv

It loads data from the legacy codebase's sites_load module (three loaders:
load_mcgann_vs, load_wotherspoon_vs, load_kaiseretal_vs for independent data,
and load_cpt_vs for clustered CPT data).

IMPORTANT: This script requires the legacy codebase environment to run:
    mamba activate oldvs30_venv
    cd /home/arr65/src/pre-refactor-Vs30-for-comparison
    python /path/to/this/script.py

The legacy sites_load module is NOT part of the refactored vs30 package.

Filtering applied:
  - Kaiser et al.: Q3 stations removed unless station name is exactly
    3 characters (broadband seismometers on rock) — same legacy filter
  - McGann and Wotherspoon: no filtering
  - CPT: no filtering

Uncertainty assignment is handled by the legacy loaders:
  - McGann: 0.2 (constant, ~20%)
  - Wotherspoon: 0.2 (constant, ~20%)
  - Kaiser et al.: variable by quality code Q: Q1=0.1, Q2=0.2, Q3=0.5
  - CPT: 0.5 (constant, ~50%)
"""

from pathlib import Path

import pandas as pd

from vs30 import sites_load

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "vs30" / "resources" / "observations"


def main():
    print("=" * 70)
    print("Exporting measured Vs30 data with source identification")
    print("=" * 70)

    # Process "original" source with individual components
    print("\nProcessing source: 'original'")
    print("-" * 70)

    # Load each source separately and add source column
    print("Loading individual data sources...")
    mcgann = sites_load.load_mcgann_vs()
    mcgann["source"] = "mcgann"

    wotherspoon = sites_load.load_wotherspoon_vs()
    wotherspoon["source"] = "wotherspoon"

    kaiseretal_unfiltered = sites_load.load_kaiseretal_vs()
    kaiseretal_unfiltered["source"] = "kaiseretal"

    # Create filtered version: remove Q3 unless station name is 3 characters
    kaiseretal_filtered = kaiseretal_unfiltered[
        (kaiseretal_unfiltered.q != 3)
        | (kaiseretal_unfiltered.station.str.len() == 3)
    ].copy()

    print(f"  McGann: {len(mcgann)} sites")
    print(f"  Wotherspoon: {len(wotherspoon)} sites")
    print(f"  Kaiser (unfiltered): {len(kaiseretal_unfiltered)} sites")
    print(f"  Kaiser (filtered): {len(kaiseretal_filtered)} sites")

    # Combine sources (filtered Kaiser et al.)
    combined = pd.concat(
        [mcgann, wotherspoon, kaiseretal_filtered], ignore_index=True
    )

    print(f"\n  Combined: {len(combined)} sites")
    print(f"  Columns: {list(combined.columns)}")

    output_file_filtered = (
        OUTPUT_DIR / "foster_2019_measured_vs30_independent_observations.csv"
    )

    print(f"\n  Writing to {output_file_filtered}...")
    combined.to_csv(output_file_filtered, index=False)
    print(f"  Exported {len(combined)} sites")

    # Process "cpt" source
    print("\nProcessing source: 'cpt'")
    print("-" * 70)

    cpt = sites_load.load_cpt_vs()
    cpt["source"] = "cpt"

    print(f"  Loaded {len(cpt)} sites")
    print(f"  Columns: {list(cpt.columns)}")

    output_file_cpt = OUTPUT_DIR / "viktor_inferred_vs30_from_cpt.csv"
    print(f"\n  Writing to {output_file_cpt}...")
    cpt.to_csv(output_file_cpt, index=False)
    print(f"  Exported {len(cpt)} sites")

    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - {output_file_filtered.name} ({len(combined)} sites)")
    print(f"  - {output_file_cpt.name} ({len(cpt)} sites)")


if __name__ == "__main__":
    main()
