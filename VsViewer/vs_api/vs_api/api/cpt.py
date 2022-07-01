import flask
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const
from VsViewer.vs_calc import CPT


@server.app.route(const.CPT_CREATE_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_cpt():
    """
    Creates a cpt and returns the result
    """
    server.app.logger.info(f"Received request at {const.CPT_CREATE_ENDPOINT}")

    (
        (cpt_json),
        optional_params_dict,
    ) = utils.get_check_keys(
        flask.request.args,
        ("cpts",),
        (),
    )

    cpt = CPT.from_json(cpt_json)


    server.app.logger.debug(
        f"Request parameters {cpt_json}"
    )

    return flask.jsonify(
                cpt.to_json(),
            )
