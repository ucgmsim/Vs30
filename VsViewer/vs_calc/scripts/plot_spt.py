import argparse
from typing import List

import matplotlib.pyplot as plt

from VsViewer.vs_calc import SPT, utils


def plot_spt(spts: List[SPT], output_ffp: str):
    """
    Plots the CPT values Qc, Fs and u at their depth values
    and saves to a given output file
    """
    fig = plt.figure(figsize=(16, 10))
    measurements = ["N", "N60"]
    plot_legend = len(spts) != 1

    for ix, measure in enumerate(measurements):
        ax = fig.add_subplot(1, 2, ix + 1)
        ax.set_xlabel(f"{measure}", size=16)
        ax.set_ylabel("Depth (m)", size=16)
        for spt_ix, spt in enumerate(spts):
            ax.plot(
                *utils.convert_to_midpoint(getattr(spt, measure), spt.depth),
                linewidth=2.5,
                label=spt.name,
            )
        if plot_legend and ix == 1:
            ax.legend(loc="upper right")
        ax.invert_yaxis()
        ax.tick_params(labelsize=15)

    plt.subplots_adjust(
        left=0.1, bottom=0.075, right=0.9, top=0.925, wspace=0.3, hspace=0.3
    )
    plt.savefig(f"{output_ffp}.png")


def main():
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spt_ffps", type=str, nargs="+", help="The full file path to the spt files"
    )
    parser.add_argument(
        "output_ffp",
        type=str,
        help="The full file path to the output file (without extension)",
    )
    args = parser.parse_args()

    # Get SPT
    spts = [SPT.from_file(spt) for spt in args.spt_ffps]

    # Plot spt
    plot_spt(spts, args.output_ffp)


if __name__ == "__main__":
    main()
