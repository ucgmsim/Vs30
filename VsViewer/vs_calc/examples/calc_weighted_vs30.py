"""
Compute the average weighted Vs30 given CPT files and correlations example
"""
from pathlib import Path

from vs_calc import CPT, VsProfile, calculate_weighted_vs30


examples_dir = Path(__file__).parent.resolve()
cpt_ffps = [examples_dir / "CPT_6457.csv", examples_dir / "CPT_6458.csv"]
cpts = [CPT.from_file(str(cpt_ffp)) for cpt_ffp in cpt_ffps]
correlations = ["andrus_2007", "mcgann_2018"]
correlation_weights = {"andrus_2007": 0.6, "mcgann_2018": 0.4}
cpt_weights = {"CPT_6457": 0.8, "CPT_6458": 0.2}

vs_profiles = [
    VsProfile.from_cpt(cpt, correlation) for cpt in cpts for correlation in correlations
]
vs30, vs30_sd = calculate_weighted_vs30(
    vs_profiles, cpt_weights, correlation_weights
)
print(f"The weighted average Vs30 is {vs30} and the Standard Deviation is {vs30_sd}")

# Expected output
# The weighted average Vs30 is 330.75434852800674 and the Standard Deviation is 67.40592850352557
