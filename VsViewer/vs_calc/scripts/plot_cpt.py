import argparse
from typing import List

import matplotlib.pyplot as plt

from VsViewer.vs_calc import CPT, utils


def plot_cpt(cpts: List[CPT], output_ffp: str):
    """
    Plots the CPT values Qc, Fs and u at their depth values
    and saves to a given output file
    """
    fig = plt.figure(figsize=(16, 10))
    measurements = ["Qc", "Fs", "u"]
    plot_legend = len(cpts) != 1

    for ix, measure in enumerate(measurements):
        ax = fig.add_subplot(1, 3, ix + 1)
        ax.set_xlabel(f"{measure} (MPa)", size=16)
        ax.set_ylabel("Depth (m)", size=16)
        for cpt_ix, cpt in enumerate(cpts):
            ax.plot(
                *utils.convert_to_midpoint(getattr(cpt, measure), cpt.depth),
                linewidth=2.5,
                label=cpt.name,
            )
        if plot_legend and ix == 2:
            ax.legend(loc="upper right")
        ax.invert_yaxis()
        ax.tick_params(labelsize=15)

    plt.subplots_adjust(
        left=0.1, bottom=0.075, right=0.9, top=0.925, wspace=0.3, hspace=0.3
    )
    plt.savefig(f"{output_ffp}.png")


def main():
    """
    Gather metadata from each realisation and outputs to a csv
    """
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cpt_ffps", type=str, nargs="+", help="The full file path to the cpt files"
    )
    parser.add_argument(
        "output_ffp",
        type=str,
        help="The full file path to the output file (without extension)",
    )
    args = parser.parse_args()

    # Get CPT
    cpts = [CPT.from_file(cpt) for cpt in args.cpt_ffps]

    # Plot cpt
    plot_cpt(cpts, args.output_ffp)

    # Print CPT info
    for cpt in cpts:
        print(f"{cpt.name} Info")
        for k, v in cpt.info.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
