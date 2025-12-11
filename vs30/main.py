import rasterio

# import typer

# # Create Typer app for CLI
# app = typer.Typer(
#     name="vs30map",
#     help="VS30 map generation and categorical model updates",
#     add_completion=False,
# )

geology_or_terrain_to_initial_raster_filename = {
    "geology": "geology.tif",
    "terrain": "/home/arr65/src/vs30/vs30/data/IwahashiPike.tif",
}


# @app.command()
def update_categorical_vs30_mean_and_standard_deviation(
    geology_or_terrain: str,
    initial_categorical_vs30_mean_and_standard_deviation_path: str,
    observations_path: str,
    output_path: str,
):
    """
    Update the categorical VS30 mean and standard deviation for a given geology or terrain model.
    """

    # if geology_or_terrain not in ["geology", "terrain"]:
    #     typer.echo(f"Error: Invalid geology or terrain: {geology_or_terrain}")
    #     raise typer.Exit(1)

    initial_raster_filename = geology_or_terrain_to_initial_raster_filename[
        geology_or_terrain
    ]

    with rasterio.open(initial_raster_filename) as src:
        raster_with_pixel_ids = src.read(1)

    print()


if __name__ == "__main__":
    # app()
    update_categorical_vs30_mean_and_standard_deviation(
        "terrain",
        "/home/arr65/src/vs30/vs30/resources/categorical_vs30_mean_and_stddev/geology/geology_model_prior_mean_and_standard_deviation.csv",
        "/home/arr65/src/vs30/vs30/resources/observations/measured_vs30_original_filtered.csv",
        "/home/arr65/src/vs30/vs30/resources/categorical_vs30_mean_and_stddev/terrain/terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
    )
