import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from VsViewer.vs_calc import CPT, VsProfile, utils, CORRELATIONS


def plot_vs_profiles(vs_profiles: List[VsProfile], output_ffp: str):
    """
    Plots the Vs Profiles with a mean and standard deviation
    and saves to a given output file
    """
    fig, (ax) = plt.subplots(1, 1, figsize=(10, 16))
    default_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ix, vs_profile in enumerate(vs_profiles):
        ax.set_xlabel(f"Vs (m/s)", size=20)
        ax.set_ylabel("Depth (m)", size=20)
        ax.plot(
            *utils.convert_to_midpoint(vs_profile.vs, vs_profile.depth),
            linewidth=2.5,
            color=default_colours[ix],
            label=f"{vs_profile.cpt.cpt_ffp.stem}_{vs_profile.correlation}",
        )
        # Standard Deviations
        ax.plot(
            *utils.convert_to_midpoint(
                vs_profile.vs * np.exp(vs_profile.vs_sd), vs_profile.depth
            ),
            linewidth=2.5,
            linestyle="dashed",
            color=default_colours[ix],
        )
        ax.plot(
            *utils.convert_to_midpoint(
                vs_profile.vs * np.exp(-vs_profile.vs_sd), vs_profile.depth
            ),
            linewidth=2.5,
            linestyle="dashed",
            color=default_colours[ix],
        )
        ax.legend(loc="upper right")
    ax.invert_yaxis()
    ax.tick_params(labelsize=15)

    plt.savefig(f"{output_ffp}.png")


def main():
    """
    Gather metadata from each realisation and outputs to a csv
    """
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpt_ffps",
        type=str,
        nargs="+",
        required=True,
        help="The full file path to the cpt files",
    )
    parser.add_argument(
        "--correlations",
        type=str,
        nargs="+",
        required=True,
        choices=CORRELATIONS.keys(),
        help="The correlation names",
    )
    parser.add_argument(
        "--output_ffp",
        type=str,
        required=True,
        help="The full file path to the output file (without extension)",
    )
    args = parser.parse_args()

    # Get CPTs Vs Profiles
    cpts = [CPT(cpt) for cpt in args.cpt_ffps]
    vs_profiles = [
        VsProfile.from_cpt(cpt, correlation)
        for cpt in cpts
        for correlation in args.correlations
    ]

    # Plot Vs Profiles
    plot_vs_profiles(vs_profiles, args.output_ffp)


if __name__ == "__main__":
    main()
