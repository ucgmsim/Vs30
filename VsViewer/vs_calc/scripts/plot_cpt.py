import argparse
import colorsys
from typing import List

from matplotlib import colors, pyplot

from VsViewer import vs_calc


def scale_saturation(color: str, scale: float):
    """
    Scales the saturation down of the color by a given percentage
    """
    rgb = colors.colorConverter.to_rgb(color)
    hls = colorsys.rgb_to_hls(*rgb)
    hls_scaled = (hls[0], hls[1], hls[2] * ((100 - scale) / 100))
    return colorsys.hls_to_rgb(*hls_scaled)


def plot_cpt(cpts: List[vs_calc.CPT], output_ffp: str):
    """
    Plots the CPT values Qc, Fs and u at their depth values
    and saves to a given output file
    """
    fig = pyplot.figure(figsize=(16, 10))
    measurements = ["Qc", "Fs", "u"]
    colours = ["Blue", "Green", "Red"]
    plot_legend = len(cpts) != 1
    scales = [i * (100 / len(cpts)) for i in range(len(cpts))]

    for ix, measure in enumerate(measurements):
        ax1 = fig.add_subplot(1, 3, ix + 1)
        ax1.set_xlabel(f"{measure} (MPa)", size=16)
        ax1.set_ylabel("Depth (m)", size=16)
        for cpt_ix, cpt in enumerate(cpts):
            pyplot.plot(
                *vs_calc.utils.convert_to_midpoint(getattr(cpt, measure), cpt.depth),
                color=scale_saturation(colours[ix], scales[cpt_ix]),
                linewidth=2.5,
                label=cpt.cpt_ffp.stem,
            )
        if plot_legend:
            ax1.legend(loc="upper right")
        pyplot.gca().invert_yaxis()
        pyplot.xticks(fontsize=13)
        pyplot.yticks(fontsize=13)

    pyplot.subplots_adjust(
        left=0.1, bottom=0.075, right=0.9, top=0.925, wspace=0.3, hspace=0.3
    )
    pyplot.savefig(f"{output_ffp}.png")


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
    cpts = [vs_calc.CPT(cpt) for cpt in args.cpt_ffps]

    # Plot cpt
    plot_cpt(cpts, args.output_ffp)

    # Print CPT info
    for cpt in cpts:
        print(f"{cpt.cpt_ffp.stem} Info")
        for k, v in cpt.info.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()