"""
Compute the average weighted Vs30 given CPT files and correlations example
"""

from pathlib import Path

from vs_calc import CPT, VsProfile, calculate_weighted_vs30

examples_dir = Path(__file__).parent.resolve()
cpt_ffps = [examples_dir / "CPT_6457.csv", examples_dir / "CPT_6458.csv"]
cpts = [CPT.from_file(str(cpt_ffp)) for cpt_ffp in cpt_ffps]
cpt_correlations = ["andrus_2007", "mcgann_2018"]
cpt_correlation_weights = {"andrus_2007": 0.6, "mcgann_2018": 0.4}
vs_weights = {"CPT_6457": 0.8, "CPT_6458": 0.2}
vs30_correlation = "boore_2004"
vs30_correlation_weights = {"boore_2004": 1.0}

vs_profiles = [
    VsProfile.from_cpt(cpt, cpt_correlation, vs30_correlation)
    for cpt in cpts
    for cpt_correlation in cpt_correlations
]
vs30, vs30_sd = calculate_weighted_vs30(
    vs_profiles, vs_weights, cpt_correlation_weights, {}, vs30_correlation_weights
)
print(f"The weighted average Vs30 is {vs30} and the Standard Deviation is {vs30_sd}")

# Expected output
# The weighted average Vs30 is 264.1903089336792 and the Standard Deviation is 4.224848228421367
