import flask
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const
from VsViewer.vs_calc.utils import convert_to_midpoint


@server.app.route(const.MIDPOINT_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def cpt_to_midpoint():
    """
    Converts CPT data to midpoint data for plotting
    """
    server.app.logger.info(f"Received request at {const.MIDPOINT_ENDPOINT}")

    json_array = flask.request.json
    cpt_dict = dict()
    for cpt_data in json_array:
        qc, depth = convert_to_midpoint(cpt_data['value']["Qc"], cpt_data['value']["depth"])
        fs, _ = convert_to_midpoint(cpt_data['value']["Fs"], cpt_data['value']["depth"])
        u, _ = convert_to_midpoint(cpt_data['value']["u"], cpt_data['value']["depth"])
        cpt_dict[cpt_data["label"]] = {
            "Depth": depth,
            "Qc": qc,
            "Fs": fs,
            "u": u,
        }
    return flask.jsonify(cpt_dict)