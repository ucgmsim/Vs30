import React, { memo, useState, useContext, useEffect } from "react";
import Select from "react-select";
import Papa from "papaparse";
import readXlsxFile from "read-excel-file";
import { wait } from "@testing-library/user-event/dist/utils";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";
import * as Utils from "Utils";

import "assets/vsProfile.css";
import {
  WeightTable,
  FileTable,
  VsProfilePreviewPlot,
  VsProfileTable,
  InfoTooltip,
} from "components";

const VsProfile = () => {
  const {
    vsProfileData,
    setVsProfileData,
    vsProfileMidpointData,
    setVsProfileMidpointData,
    vsProfileWeights,
    setVsProfileWeights,
    setVsProfileResults,
  } = useContext(GlobalContext);

  // VsProfile Table
  const [vsProfileInfo, setVsProfileInfo] = useState(null);
  const [vsProfileTableData, setVsProfileTableData] = useState({});
  const [selectedVsProfileTable, setSelectedVsProfileTable] = useState(null);
  const [selectedVsProfileTableData, setSelectedVsProfileTableData] =
    useState(null);
  // VsProfile Plot
  const [selectedVsProfilePlot, setSelectedVsProfilePlot] = useState(null);
  const [vsProfilePlotData, setVsProfilePlotData] = useState({});
  const [vsProfileAveragePlotData, setVsProfileAveragePlotData] = useState({});
  // Form variables
  const [file, setFile] = useState("");
  const [vsProfileName, setVsProfileName] = useState("");
  const [layered, setLayered] = useState(false);
  const [VsProfileOptions, setVsProfileOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [canSet, setCanSet] = useState(false);
  const [canAdd, setCanAdd] = useState(false);
  // Errors
  const [flashWeightError, setFlashWeightError] = useState(false);
  const [weightError, setWeightError] = useState(false);
  const [flashFileUploadError, setFlashFileUploadError] = useState(false);
  const [flashNameUploadError, setFlashNameUploadError] = useState(false);
  const [flashServerError, setFlashServerError] = useState(false);
  const [uploadError, setUploadError] = useState(false);
  const [uploadErrorText, setUploadErrorText] = useState(CONSTANTS.FILE_ERROR);

  // Set the VsProfile Weights
  useEffect(() => {
    if (VsProfileOptions.length > 0) {
      let tempVsWeights = {};
      VsProfileOptions.forEach((entry) => {
        tempVsWeights[entry["label"]] = 1 / VsProfileOptions.length;
      });
      setVsProfileWeights(tempVsWeights);
      setCanSet(true);
    } else {
      setCanSet(false);
    }
  }, [VsProfileOptions]);

  const sendProcessRequest = async () => {
    if (vsProfileName in vsProfileData) {
      setUploadError(true);
      setUploadErrorText(CONSTANTS.NAME_ERROR);
      setFlashNameUploadError(true);
      await wait(1000);
      setFlashNameUploadError(false);
    } else {
      setUploadError(false);
      setLoading(true);
      let serverResponse = false;
      const formData = new FormData();
      formData.append(file.name, file);
      formData.append(
        file.name + "_formData",
        JSON.stringify({
          vsProfileName: vsProfileName,
          layered: layered ? "True" : "False",
        })
      );
      await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_CREATE_ENDPOINT, {
        method: "POST",
        body: formData,
      })
        .then(async (response) => {
          if (response.ok) {
            serverResponse = true;
            const responseData = await response.json();
            // Set Table Data
            let tempVsTableData = vsProfileTableData;
            if (file.name.split(".")[1] === "xlsx") {
              readXlsxFile(file).then((rows) => {
                rows.shift();
                tempVsTableData[vsProfileName] = rows;
              });
            } else {
              Papa.parse(file, {
                header: true,
                skipEmptyLines: true,
                complete: function (results, file) {
                  tempVsTableData[vsProfileName] = results.data;
                },
              });
            }
            setVsProfileTableData(tempVsTableData);
            // Add to VsProfile Select Dropdown and VsProfile Data
            let tempOptions = [];
            let tempVsData = vsProfileData;
            for (const sptOption of VsProfileOptions) {
              tempOptions.push({
                value: sptOption["value"],
                label: sptOption["label"],
              });
            }
            for (const key of Object.keys(responseData)) {
              tempOptions.push({
                value: responseData[key],
                label: responseData[key]["name"],
              });
              tempVsData[key] = responseData[key];
            }
            setVsProfileOptions(tempOptions);
            setVsProfileData(tempVsData);
          } else {
            setUploadErrorText(CONSTANTS.FILE_ERROR);
            setUploadError(true);
            setFlashFileUploadError(true);
            await wait(1000);
            setFlashFileUploadError(false);
          }
          setLoading(false);
        })
        .catch(async () => {
          if (serverResponse) {
            setUploadErrorText(CONSTANTS.FILE_ERROR);
            setUploadError(true);
            setFlashFileUploadError(true);
            await wait(1000);
            setFlashFileUploadError(false);
          } else {
            setUploadErrorText(CONSTANTS.REQUEST_ERROR);
            setUploadError(true);
            setFlashServerError(true);
            await wait(1000);
            setFlashServerError(false);
          }
          setLoading(false);
        });
    }
  };

  const sendVsProfileMidpointRequest = async (vsProfileToSend) => {
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_MIDPOINT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(vsProfileToSend),
    }).then(async (response) => {
      const responseData = await response.json();
      // Add to MidpointData
      let tempMidpointData = vsProfileMidpointData;
      for (const key of Object.keys(responseData)) {
        tempMidpointData[key] = responseData[key];
      }
      setVsProfileMidpointData(tempMidpointData);
    });
  };

  const sendAverageRequest = async () => {
    let vsProfilesToSend = {};
    for (const key of Object.keys(vsProfileData)) {
      vsProfilesToSend[key] = vsProfileData[key];
    }
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_AVERAGE_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vsProfiles: vsProfilesToSend,
        vsWeights: vsProfileWeights,
        vs30CorrelationWeights: {},
        vsCorrelationWeights: {},
      }),
    }).then(async (response) => {
      const responseData = await response.json();
      // Set the Plot Average Data
      setVsProfileAveragePlotData(responseData["average"]);
    });
  };

  const changeVsProfileSelection = async (entries) => {
    // Gather Midpoint data
    let VsProfilesToSend = [];
    let VsProfileLabels = [];
    entries.forEach((entry) => {
      VsProfileLabels.push(entry["label"]);
      if (!vsProfileMidpointData.hasOwnProperty(entry["label"])) {
        VsProfilesToSend.push(entry["value"]);
      }
    });
    if (VsProfilesToSend.length !== 0) {
      await sendVsProfileMidpointRequest(VsProfilesToSend);
    }
    let tempPlotData = {};
    // Create VsProfile Plot Data
    VsProfileLabels.forEach((name) => {
      tempPlotData[name] = vsProfileMidpointData[name];
    });
    setVsProfilePlotData(tempPlotData);
    setSelectedVsProfilePlot(entries);
  };

  const onSelectVsProfileTable = (e) => {
    setSelectedVsProfileTable(e);
    setSelectedVsProfileTableData(vsProfileTableData[e["label"]]);
    setVsProfileInfo(vsProfileData[e["label"]]["info"]);
  };

  const onSetFile = (e) => {
    setFile(e);
    setVsProfileName(e.name.split(".")[0]);
    setCanAdd(true);
  };

  const checkWeights = async () => {
    let check = Utils.errorCheckWeights(vsProfileWeights);
    if (check) {
      let tempVsResults = [];
      Object.keys(vsProfileData).forEach(function (key) {
        tempVsResults.push({ label: key, value: vsProfileData[key] });
      });
      setVsProfileResults(tempVsResults);
      // Remove average for now
      // sendAverageRequest();
      // Ensures the values are floats
      Object.keys(vsProfileWeights).forEach(function (key) {
        vsProfileWeights[key] = parseFloat(vsProfileWeights[key]);
      });
      setWeightError(false);
    } else {
      setFlashWeightError(true);
      setWeightError(true);
      await wait(1000);
      setFlashWeightError(false);
    }
  };

  // Change the vsProfile Weights
  const changeVsProfileWeights = (newWeights) => {
    setVsProfileWeights(newWeights);
  };

  const removeFile = (fileToRemove) => {
    let newVsOptions = [];
    let newVsData = {};
    let newPlotSelected = [];
    let newPlotData = {};
    let newVsProfileMidpointData = {};
    VsProfileOptions.forEach((object) => {
      if (object["label"] !== fileToRemove["label"]) {
        newVsOptions.push(object);
        newVsData[object["label"]] = vsProfileData[object["label"]];
        newPlotSelected.push(object);
        if (vsProfilePlotData[object["label"]] !== undefined) {
          newPlotData[object["label"]] = vsProfilePlotData[object["label"]];
          newVsProfileMidpointData[object["label"]] =
            vsProfileMidpointData[object["label"]];
        }
      }
    });
    setSelectedVsProfilePlot(newPlotSelected);
    setVsProfilePlotData(newPlotData);
    setVsProfileData(newVsData);
    setVsProfileOptions(newVsOptions);
    changeVsProfileWeights(newVsOptions);
    setVsProfileMidpointData(newVsProfileMidpointData);
    if (selectedVsProfileTable === fileToRemove) {
      setSelectedVsProfileTable(null);
    }
    setVsProfileAveragePlotData({});
  };

  return (
    <div>
      <div className="row three-column-row vs-top">
        <div className="col-1 center-elm vs-left-panel">
          <div className="vs-upload-title">Upload VsProfile</div>
          <div className="outline add-vs">
            <div
              className={
                flashFileUploadError
                  ? "cpt-flash-warning row two-colum-row vs-file-input-section"
                  : "row two-colum-row vs-file-input-section temp-border"
              }
            >
              <input
                className="col-8 vs-file-input"
                type="file"
                onChange={(e) => onSetFile(e.target.files[0])}
              />
              <div className="col-1 file-info">
                <InfoTooltip text={CONSTANTS.VS_FILE} />
              </div>
            </div>
            <div className="form-label">VsProfile Name</div>
            <div className="stretch">
              <input
                className={
                  flashNameUploadError
                    ? "cpt-flash-warning text-input"
                    : "cpt-input text-input"
                }
                value={vsProfileName}
                onChange={(e) => setVsProfileName(e.target.value)}
              />
            </div>
            <div className="row two-colum-row layered-section center-elm">
              <div className="col-8 layered-title">Layered Approach</div>
              <input
                className="col-1 vs-checkbox"
                type="checkbox"
                onChange={(e) => setLayered(e.target.checked)}
              />
            </div>
            <div className="row two-colum-row add-vs-btn-section temp-border">
              <button
                disabled={loading || !canAdd}
                className={
                  flashServerError
                    ? "trans-btn form btn btn-danger add-vs-btn"
                    : "trans-btn add-vs-btn form btn btn-primary"
                }
                onClick={() => sendProcessRequest()}
              >
                Add VsProfile
              </button>
              <div className="col-1 weight-error">
                {uploadError && (
                  <InfoTooltip text={uploadErrorText} error={true} />
                )}
              </div>
            </div>
          </div>
          <div className="vs-file-section center-elm">
            <div className="form-section-title">VsProfile Files</div>
            <div className="file-table-section outline form center-elm">
              {Object.keys(VsProfileOptions).length > 0 && (
                <FileTable
                  files={VsProfileOptions}
                  removeFunction={removeFile}
                ></FileTable>
              )}
            </div>
          </div>
          <div className="vs-weight-title">VsProfile Weights</div>
          <div className="outline center-elm vs-weights">
            <div className="vs-weight-area">
              {Object.keys(vsProfileWeights).length > 0 && (
                <WeightTable
                  weights={vsProfileWeights}
                  setFunction={changeVsProfileWeights}
                  flashError={flashWeightError}
                />
              )}
            </div>
            <div className="row two-colum-row set-weights-section">
              <button
                disabled={!canSet}
                className="col-5 set-weights preview-btn btn btn-primary"
                onClick={() => checkWeights()}
              >
                Set Weights
              </button>
              <div className="col-1 weight-error">
                {weightError && (
                  <InfoTooltip text={CONSTANTS.WEIGHT_ERROR} error={true} />
                )}
              </div>
            </div>
          </div>
        </div>
        <div className="col-2 center-elm vs-table-section">
          <div className="center-elm">
            <div className="vs-table-title">VsProfile Table</div>
            <Select
              className="select-box"
              placeholder="Select VsProfiles"
              options={VsProfileOptions}
              isDisabled={VsProfileOptions.length === 0}
              value={selectedVsProfileTable}
              onChange={(e) => onSelectVsProfileTable(e)}
            ></Select>
          </div>
          <div className="outline vs-table">
            {Object.keys(vsProfileData).length > 0 &&
              selectedVsProfileTable !== null && (
                <VsProfileTable
                  vsProfileData={selectedVsProfileTableData}
                  vsProfileInfo={vsProfileInfo}
                />
              )}
          </div>
        </div>
        <div className="col-3 center-elm vs-plot-section">
          <div className="center-elm">
            <div className="vs-plot-title">VsProfile Plot</div>
            <Select
              className="select-box"
              placeholder="Select VsProfiles"
              isMulti={true}
              options={VsProfileOptions}
              isDisabled={VsProfileOptions.length === 0}
              value={selectedVsProfilePlot}
              onChange={(e) => changeVsProfileSelection(e)}
            ></Select>
          </div>
          <div className="outline vs-plot">
            {Object.keys(vsProfilePlotData).length > 0 && (
              <VsProfilePreviewPlot
                className="vs-plot"
                vsProfilePlotData={vsProfilePlotData}
                average={vsProfileAveragePlotData}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(VsProfile);
