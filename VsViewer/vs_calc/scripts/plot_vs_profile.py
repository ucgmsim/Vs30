import argparse
from typing import List

import numpy as np
from matplotlib import pyplot

from VsViewer import vs_calc


def plot_vs_profiles(vs_profiles: List[vs_calc.VsProfile], output_ffp: str):
    """
    Plots the Vs Profiles with a mean and standard deviation
    and saves to a given output file
    """
    fig, (ax1) = pyplot.subplots(1, 1, figsize=(10, 16))
    default_colours = pyplot.rcParams["axes.prop_cycle"].by_key()["color"]

    for ix, vs_profile in enumerate(vs_profiles):
        ax1.set_xlabel(f"Vs (m/s)", size=20)
        ax1.set_ylabel("Depth (m)", size=20)
        pyplot.plot(
            *vs_calc.utils.convert_to_midpoint(vs_profile.vs, vs_profile.depth),
            linewidth=2.5,
            color=default_colours[ix],
            label=f"{vs_profile.cpt.cpt_ffp.stem}_{vs_profile.correlation}",
        )
        # Standard Deviations
        pyplot.plot(
            *vs_calc.utils.convert_to_midpoint(
                vs_profile.vs * np.exp(vs_profile.vs_sd), vs_profile.depth
            ),
            linewidth=2.5,
            linestyle="dashed",
            color=default_colours[ix],
        )
        pyplot.plot(
            *vs_calc.utils.convert_to_midpoint(
                vs_profile.vs * np.exp(-vs_profile.vs_sd), vs_profile.depth
            ),
            linewidth=2.5,
            linestyle="dashed",
            color=default_colours[ix],
        )
        ax1.legend(loc="upper right")
    pyplot.gca().invert_yaxis()
    pyplot.xticks(fontsize=15)
    pyplot.yticks(fontsize=15)

    pyplot.savefig(f"{output_ffp}.png")


def main():
    """
    Gather metadata from each realisation and outputs to a csv
    """
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cpt_ffps", type=str, nargs="+", help="The full file path to the cpt files"
    )
    parser.add_argument(
        "-correlations", type=str, nargs="+", help="The correlation names"
    )
    parser.add_argument(
        "-output_ffp",
        type=str,
        help="The full file path to the output file (without extension)",
    )
    args = parser.parse_args()

    # Get CPTs Vs Profiles
    cpts = [vs_calc.CPT(cpt) for cpt in args.cpt_ffps]
    vs_profiles = [
        vs_calc.VsProfile(cpt, correlation) for cpt in cpts for correlation in args.correlations
    ]

    # Plot Vs Profiles
    plot_vs_profiles(vs_profiles, args.output_ffp)


if __name__ == "__main__":
    main()