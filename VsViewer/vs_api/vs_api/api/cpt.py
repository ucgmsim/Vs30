import flask
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const
from VsViewer.vs_calc import CPT
from VsViewer.vs_calc.constants import CORRELATIONS


@server.app.route(const.CPT_CREATE_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_cpts():
    """
    Creates cpts and returns the result
    """
    server.app.logger.info(f"Received request at {const.CPT_CREATE_ENDPOINT}")

    csvs = flask.request.files
    cpt_dict = dict()
    for csv_name, csv_data in csvs.items():
        cpt = CPT.from_byte_stream(csv_data.filename, csv_data.stream.read())
        cpt_dict[cpt.name] = cpt.to_json()

    return flask.jsonify(cpt_dict)


@server.app.route(const.GET_CORRELATIONS_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def get_correlations():
    """
    Gets the currently supported correlations for CPT to Vs
    """
    server.app.logger.info(f"Received request at {const.GET_CORRELATIONS_ENDPOINT}")
    return flask.jsonify(list(CORRELATIONS.keys()))
