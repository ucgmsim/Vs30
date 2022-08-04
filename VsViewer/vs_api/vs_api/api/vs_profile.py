import flask
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const
from VsViewer.vs_calc import CPT
from VsViewer.vs_calc import SPT
from VsViewer.vs_calc import VsProfile


@server.app.route(const.VSPROFILE_FROM_CPT_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_vsprofile_from_cpt():
    """
    Creates a VsProfile from a set of cpts and set of correlations
    """
    server.app.logger.info(f"Received request at {const.VSPROFILE_FROM_CPT_ENDPOINT}")

    json = flask.request.json
    response_dict = dict()
    for cpt_name, cpt_json in json["cpts"].items():
        cpt = CPT.from_json(cpt_json)
        for correlation in json["correlations"]:
            vs_profile = VsProfile.from_cpt(cpt, correlation)
            response_dict[
                f"{vs_profile.name}_{vs_profile.correlation}"
            ] = vs_profile.to_json()
    return flask.jsonify(response_dict)


@server.app.route(const.VSPROFILE_FROM_SPT_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_vsprofile_from_spt():
    """
    Creates a VsProfile from a set of spts and set of correlations
    """
    server.app.logger.info(f"Received request at {const.VSPROFILE_FROM_SPT_ENDPOINT}")

    json = flask.request.json
    response_dict = dict()
    for spt_name, spt_json in json["spts"].items():
        spt = SPT.from_json(spt_json)
        for correlation in json["correlations"]:
            vs_profile = VsProfile.from_spt(spt, correlation)
            response_dict[
                f"{vs_profile.name}_{vs_profile.correlation}"
            ] = vs_profile.to_json()
    return flask.jsonify(response_dict)
