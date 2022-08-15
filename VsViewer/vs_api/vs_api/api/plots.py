import flask
import numpy as np
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const
from vs_calc.SPT import SPT
from vs_calc.utils import convert_to_midpoint


@server.app.route(const.CPT_MIDPOINT_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def cpt_to_midpoint():
    """
    Converts CPT data to midpoint data for plotting
    """
    server.app.logger.info(f"Received request at {const.CPT_MIDPOINT_ENDPOINT}")

    json_array = flask.request.json
    cpt_dict = dict()
    for cpt_data in json_array:
        qc, depth = convert_to_midpoint(
            cpt_data["value"]["Qc"], cpt_data["value"]["depth"]
        )
        fs, _ = convert_to_midpoint(cpt_data["value"]["Fs"], cpt_data["value"]["depth"])
        u, _ = convert_to_midpoint(cpt_data["value"]["u"], cpt_data["value"]["depth"])
        cpt_dict[cpt_data["label"]] = {
            "Depth": depth,
            "Qc": qc,
            "Fs": fs,
            "u": u,
        }
    return flask.jsonify(cpt_dict)


@server.app.route(const.VS_PROFILE_MIDPOINT_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def vs_profile_to_midpoint():
    """
    Converts VsProfile data to midpoint data for plotting
    """
    server.app.logger.info(f"Received request at {const.VS_PROFILE_MIDPOINT_ENDPOINT}")

    json_array = flask.request.json
    vs_profile_dict = dict()
    for vs_profile_data in json_array:
        vs, depth = convert_to_midpoint(vs_profile_data["vs"], vs_profile_data["depth"])
        vs_sd_below, _ = convert_to_midpoint(
            np.asarray(vs_profile_data["vs"])
            * np.exp(-np.asarray(vs_profile_data["vs_sd"])),
            vs_profile_data["depth"],
        )
        vs_sd_above, _ = convert_to_midpoint(
            np.asarray(vs_profile_data["vs"])
            * np.exp(np.asarray(vs_profile_data["vs_sd"])),
            vs_profile_data["depth"],
        )
        vs_profile_name = (
            vs_profile_data["name"]
            if vs_profile_data["correlation"] == ""
            else f"{vs_profile_data['name']}_{vs_profile_data['correlation']}"
        )
        vs_profile_dict[vs_profile_name] = {
            "Depth": depth,
            "Vs": vs,
            "VsSDBelow": vs_sd_below,
            "VsSDAbove": vs_sd_above,
        }
    return flask.jsonify(vs_profile_dict)


@server.app.route(const.SPT_MIDPOINT_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def spt_to_midpoint():
    """
    Converts SPT data to midpoint data for plotting
    """
    server.app.logger.info(f"Received request at {const.SPT_MIDPOINT_ENDPOINT}")

    json_array = flask.request.json
    spt_dict = dict()
    for spt_data in json_array:
        spt = SPT.from_json(spt_data["value"])
        n, depth = convert_to_midpoint(spt.N, spt.depth)
        n60, _ = convert_to_midpoint(spt.N60, spt.depth)
        spt_dict[spt_data["label"]] = {
            "Depth": depth,
            "N": n,
            "N60": n60,
        }
    return flask.jsonify(spt_dict)
