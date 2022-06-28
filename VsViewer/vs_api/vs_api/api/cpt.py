import flask
from flask_cors import cross_origin

from vs_api import server

@server.app.route(const.ENSEMBLE_DISAGG_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@au.api.endpoint_exception_handling(server.app)
def get_ensemble_disagg():
    """Retrieves the contribution of each rupture for the
    specified exceedance.

    Valid request has to contain the following
    URL parameters: ensemble_id, station, im, exceedance
    """
    server.app.logger.info(f"Received request at {const.ENSEMBLE_DISAGG_ENDPOINT}")
    cache = server.cache

    (
        (ensemble_id, station, im, exceedance,),
        optional_params_dict,
    ) = au.api.get_check_keys(
        flask.request.args,
        ("ensemble_id", "station", "im", "exceedance"),
        (("gmt_plot", bool, False), ("vs30", float), ("im_component", str, "RotD50"),),
    )

    gmt_plots = optional_params_dict["gmt_plot"]
    user_vs30 = optional_params_dict.get("vs30")
    im = sc.im.IM.from_str(im, im_component=optional_params_dict.get("im_component"))

    server.app.logger.debug(
        f"Request parameters {ensemble_id}, {station}, {im}, {im.component}, {exceedance}"
    )


    return flask.jsonify(
                {
                    "type": "ensemble_disagg",
                    "ensemble_id": ensemble_id,
                    "station": station,
                    "im": str(im),
                    "im_component": str(im.component),
                    "exceedance": exceedance,
                    "gmt_plots": gmt_plots,
                    "user_vs30": user_vs30,
                },
            )