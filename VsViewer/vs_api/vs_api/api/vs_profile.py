import json
import flask
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const
from vs_calc import CPT
from vs_calc import SPT
from vs_calc import VsProfile
from vs_calc import VS30_CORRELATIONS
from vs_calc.calc_weightings import calculate_weighted_vs30


@server.app.route(const.VS_PROFILE_CREATE_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_vsprofile():
    """
    Creates a VsProfile from a file and form data
    """
    server.app.logger.info(f"Received request at {const.VS_PROFILE_CREATE_ENDPOINT}")

    files = flask.request.files
    vs_profile_dict = dict()
    for file_name, file_data in files.items():
        form_data = json.loads(flask.request.form.get(f"{file_name}_formData"))
        vs_profile = VsProfile.from_byte_stream(
            file_name,
            form_data.get("vsProfileName"),
            form_data.get("layered") == "True",
            file_data.stream.read(),
        )
        for depth in vs_profile.depth:
            if depth < 0:
                raise ValueError("Depth can't be negative")
        vs_profile_dict[vs_profile.name] = vs_profile.to_json()
    return flask.jsonify(vs_profile_dict)


@server.app.route(const.VS_PROFILE_FROM_CPT_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_vsprofile_from_cpt():
    """
    Creates a VsProfile from a set of cpts and set of correlations
    """
    server.app.logger.info(f"Received request at {const.VS_PROFILE_FROM_CPT_ENDPOINT}")

    json = flask.request.json
    response_dict = dict()
    for cpt_name, cpt_json in json["cpts"].items():
        cpt = CPT.from_json(cpt_json)
        for correlation in json["correlations"]:
            vs_profile = VsProfile.from_cpt(cpt, correlation)
            response_dict[
                f"{vs_profile.name}_{vs_profile.vs_correlation}"
            ] = vs_profile.to_json()
    return flask.jsonify(response_dict)


@server.app.route(const.VS_PROFILE_FROM_SPT_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_vsprofile_from_spt():
    """
    Creates a VsProfile from a set of spts and set of correlations
    """
    server.app.logger.info(f"Received request at {const.VS_PROFILE_FROM_SPT_ENDPOINT}")

    json = flask.request.json
    response_dict = dict()
    for spt_name, spt_json in json["spts"].items():
        spt = SPT.from_json(spt_json)
        for correlation in json["correlations"]:
            vs_profile = VsProfile.from_spt(spt, correlation)
            response_dict[
                f"{vs_profile.name}_{vs_profile.vs_correlation}"
            ] = vs_profile.to_json()
    return flask.jsonify(response_dict)


@server.app.route(const.VS_PROFILE_CORRELATIONS_ENDPOINT, methods=["GET"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def get_vs_profile_correlations():
    """
    Gets the currently supported correlations for VsZ to Vs30
    """
    server.app.logger.info(
        f"Received request at {const.VS_PROFILE_CORRELATIONS_ENDPOINT}"
    )
    return flask.jsonify(list(VS30_CORRELATIONS.keys()))


@server.app.route(const.VS_PROFILE_VS30_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def compute_vs30():
    """
    Computes the Vs30 based on VsProfiles and weights
    """
    server.app.logger.info(f"Received request at {const.VS_PROFILE_VS30_ENDPOINT}")
    json = flask.request.json
    # Create the VsProfiles
    vs_profiles = []
    for correlation in json["vs30CorrelationWeights"]:
        for vs_profile in json["vsProfiles"].values():
            vs_profile["vs30_correlation"] = correlation
            vs_profiles.append(VsProfile.from_json(vs_profile))
    vs30, vs30_sd = calculate_weighted_vs30(
        vs_profiles,
        json["vsWeights"],
        json["cptVsCorrelationWeights"],
        json["sptVsCorrelationWeights"],
        json["vs30CorrelationWeights"],
    )
    return flask.jsonify({"Vs30": vs30, "Vs30_SD": vs30_sd})
