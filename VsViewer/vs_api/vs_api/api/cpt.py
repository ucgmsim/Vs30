import json
import flask
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const
from vs_calc import CPT
from vs_calc.cpt_vs_correlations import CPT_CORRELATIONS


@server.app.route(const.CPT_CREATE_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_cpts():
    """
    Creates cpts and returns the result
    """
    server.app.logger.info(f"Received request at {const.CPT_CREATE_ENDPOINT}")

    files = flask.request.files
    cpt_dict = dict()
    for file_name, file_data in files.items():
        form_data = json.loads(flask.request.form.get(f"{file_name}_formData"))
        cpt = CPT.from_byte_stream(file_name, file_data.stream.read(), form_data)
        if any(cpt.depth < 0):
            raise ValueError("Depth can't be negative")
        cpt_dict[cpt.name] = cpt.to_json()
    return flask.jsonify(cpt_dict)


@server.app.route(const.GET_CPT_CORRELATIONS_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def get_cpt_correlations():
    """
    Gets the currently supported correlations for CPT to Vs
    """
    server.app.logger.info(f"Received request at {const.GET_CPT_CORRELATIONS_ENDPOINT}")
    return flask.jsonify(list(CPT_CORRELATIONS.keys()))
