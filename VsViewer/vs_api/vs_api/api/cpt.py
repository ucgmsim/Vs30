import flask
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const


@server.app.route(const.CPT_CREATE_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.api.endpoint_exception_handling(server.app)
def create_cpt():
    """
    Creates a cpt and returns the result
    """
    server.app.logger.info(f"Received request at {const.CPT_CREATE_ENDPOINT}")

    (
        (cpts),
        optional_params_dict,
    ) = utils.api.get_check_keys(
        flask.request.args,
        ("cpts"),
        (),
    )

    server.app.logger.debug(
        f"Request parameters {cpts}"
    )

    return flask.jsonify(
                {
                    "Worked": True
                },
            )