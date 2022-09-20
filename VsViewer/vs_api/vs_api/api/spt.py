import flask
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const
from vs_calc import SPT
from vs_calc import constants as vs_calc_constants
from vs_calc.spt_vs_correlations import SPT_CORRELATIONS


@server.app.route(const.SPT_CREATE_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_spt():
    """
    Creates a spt and returns the result
    """
    server.app.logger.info(f"Received request at {const.SPT_CREATE_ENDPOINT}")

    files = flask.request.files
    spt_dict = dict()
    for file_name, file_data in files.items():
        form_data = eval(flask.request.form.get(f"{file_name}_formData"))
        spt = SPT.from_byte_stream_form(file_name, file_data.stream.read(), form_data)
        for depth in spt.depth:
            if depth < 0:
                raise ValueError("Depth can't be negative")
        spt_dict[spt.name] = spt.to_json()
    return flask.jsonify(spt_dict)


@server.app.route(const.GET_SPT_CORRELATIONS_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def get_spt_correlations():
    """
    Gets the currently supported correlations for SPT to Vs
    """
    server.app.logger.info(f"Received request at {const.GET_SPT_CORRELATIONS_ENDPOINT}")
    return flask.jsonify(list(SPT_CORRELATIONS.keys()))


@server.app.route(const.GET_HAMMER_TYPES_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def get_hammer_types():
    """
    Gets the currently supported correlations for SPT to Vs
    """
    server.app.logger.info(f"Received request at {const.GET_HAMMER_TYPES_ENDPOINT}")
    return flask.jsonify([x.name for x in vs_calc_constants.HammerType])


@server.app.route(const.GET_SOIL_TYPES_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def get_soil_types():
    """
    Gets the currently supported correlations for SPT to Vs
    """
    server.app.logger.info(f"Received request at {const.GET_SOIL_TYPES_ENDPOINT}")
    return flask.jsonify([x.name for x in vs_calc_constants.SoilType])

