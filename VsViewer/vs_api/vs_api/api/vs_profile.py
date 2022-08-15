import flask
from flask_cors import cross_origin

from vs_api import server, utils
from vs_api import constants as const
from VsViewer.vs_calc import CPT
from VsViewer.vs_calc import SPT
from VsViewer.vs_calc import VsProfile
from VsViewer.vs_calc import VS_PROFILE_CORRELATIONS
from VsViewer.vs_calc import weighted_vs30


@server.app.route(const.VS_PROFILE_CREATE_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def create_vsprofile():
    """
    Creates a VsProfile from a file and form data
    """
    server.app.logger.info(f"Received request at {const.VS_PROFILE_CREATE_ENDPOINT}")

    csvs = flask.request.files
    vs_profile_dict = dict()
    for csv_name, csv_data in csvs.items():
        form_data = eval(flask.request.form.get(f"{csv_name}_formData"))
        vs_profile = VsProfile.from_byte_stream(
            form_data.get("vsProfileName"), csv_data.stream.read()
        )
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
                f"{vs_profile.name}_{vs_profile.correlation}"
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
                f"{vs_profile.name}_{vs_profile.correlation}"
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
    return flask.jsonify(list(VS_PROFILE_CORRELATIONS.keys()))


@server.app.route(const.VS_PROFILE_VS30_ENDPOINT, methods=["POST"])
@cross_origin(expose_headers=["Content-Type", "Authorization"])
@utils.endpoint_exception_handling(server.app)
def compute_vs30():
    """
    Computes the Vs30 based on VsProfiles and weights
    """
    server.app.logger.info(f"Received request at {const.VS_PROFILE_VS30_ENDPOINT}")
    json = flask.request.json
    vs30, vs30_sd = weighted_vs30.calculate_weighted_vs30(
        [VsProfile.from_json(vs_profile["value"]) for vs_profile in json["vsProfiles"]],
        json["vsWeights"],
        json["correlationWeights"],
    )
    return flask.jsonify({"Vs30": vs30, "Vs30_SD": vs30_sd})
