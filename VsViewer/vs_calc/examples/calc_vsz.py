"""
Compute the vsZ value given a CPT file and correlation example
"""
from pathlib import Path

from vs_calc import CPT, VsProfile


examples_dir = Path(__file__).parent.resolve()
cpt_ffp = examples_dir / "CPT_6457.csv"
cpt = CPT.from_file(str(cpt_ffp))
cpt_correlation = "andrus_2007"
vs_profile = VsProfile.from_cpt(cpt, cpt_correlation)

print(f"VsZ for the VsProfile {vs_profile.name}_{vs_profile.vs_correlation} is {vs_profile.vsz} at Z depth of {vs_profile.max_depth}m")

# Expected output
# VsZ for the VsProfile CPT_6457_andrus_2007 is 170.37157755399687 at Z depth of 10m
