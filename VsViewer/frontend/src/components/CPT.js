import React, { useEffect, useState, useContext, memo } from "react";
import Select from "react-select";
import Papa from "papaparse";
import readXlsxFile from "read-excel-file";
import { wait } from "@testing-library/user-event/dist/utils";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";
import * as Utils from "Utils";

import "assets/cpt.css";
import {
  CPTPlot,
  CptTable,
  WeightTable,
  FileTable,
  InfoTooltip,
  VsProfilePreviewPlot,
} from "components";
import { Switch } from "@mui/material";

const CPT = () => {
  const {
    cptData,
    setCPTData,
    cptMidpointData,
    setCptMidpointData,
    vsProfileMidpointData,
    setVsProfileMidpointData,
    cptWeights,
    setCptWeights,
    setCptResults,
    setCptCorrelationWeights,
  } = useContext(GlobalContext);

  // CPT Plot
  const [selectedCptPlot, setSelectedCptPlot] = useState(null);
  const [cptPlotData, setCptPlotData] = useState({});
  // CPT Table
  const [cptTableData, setCptTableData] = useState({});
  const [selectedCptTable, setSelectedCptTable] = useState(null);
  const [selectedCptTableData, setSelectedCptTableData] = useState(null);
  const [cptInfo, setCptInfo] = useState(null);
  const [dataIsKPa, setDataIsKPa] = useState(false);
  // VsProfilePreview
  const [vsProfileData, setVsProfileData] = useState({});
  const [vsProfilePlotData, setVsProfilePlotData] = useState({});
  const [vsProfileAveragePlotData, setVsProfileAveragePlotData] = useState({});
  // Form variables
  const [file, setFile] = useState("");
  const [cptName, setCptName] = useState("");
  const [iskPa, setiskPa] = useState(false);
  const [gwl, setGWL] = useState(1);
  const [nar, setNAR] = useState(0.8);
  const [cptOptions, setCPTOptions] = useState([]);
  const [correlationsOptions, setCorrelationsOptions] = useState([]);
  const [selectedCorrelations, setSelectedCorrelations] = useState([]);
  const [correlationWeights, setCorrelationWeights] = useState({});
  const [loading, setLoading] = useState(false);
  const [canSet, setCanSet] = useState(false);
  // Errors
  const [flashCPTWeightError, setFlashCPTWeightError] = useState(false);
  const [flashCorWeightError, setFlashCorWeightError] = useState(false);
  const [weightError, setWeightError] = useState(false);
  const [flashFileUploadError, setFlashFileUploadError] = useState(false);
  const [flashNameUploadError, setFlashNameUploadError] = useState(false);
  const [flashGWLUploadError, setFlashGWLUploadError] = useState(false);
  const [flashNARUploadError, setFlashNARUploadError] = useState(false);
  const [flashServerError, setFlashServerError] = useState(false);
  const [uploadError, setUploadError] = useState(false);
  const [uploadErrorText, setUploadErrorText] = useState(CONSTANTS.FILE_ERROR);

  // Set the Correlation Weights
  useEffect(() => {
    let tempCorWeights = {};
    let tempNewVsData = {};
    selectedCorrelations.forEach((entry) => {
      tempCorWeights[entry["label"]] = 1 / selectedCorrelations.length;
      cptOptions.forEach((object) => {
        let key = object["label"] + "_" + entry["label"];
        tempNewVsData[key] = vsProfileData[key];
      });
    });
    setVsProfileData(tempNewVsData);
    setCorrelationWeights(tempCorWeights);
  }, [selectedCorrelations]);

  // Check the user can set Weights
  useEffect(() => {
    if (selectedCorrelations.length > 0 && Object.keys(cptWeights).length > 0) {
      setCanSet(true);
    } else {
      setCanSet(false);
    }
  }, [selectedCorrelations, cptWeights]);

  // Get Correlations on page load
  useEffect(() => {
    if (correlationsOptions.length === 0) {
      fetch(CONSTANTS.VS_API_URL + CONSTANTS.GET_CPT_CORRELATIONS_ENDPOINT, {
        method: "GET",
      }).then(async (response) => {
        const responseData = await response.json();
        // Set Correlation Select Dropdown
        let tempOptionArray = [];
        for (const value of Object.values(responseData)) {
          tempOptionArray.push({ value: value, label: value });
        }
        setCorrelationsOptions(tempOptionArray);
      });
    }
  }, []);

  const changeCptWeights = (cptArray) => {
    let tempCptWeights = {};
    cptArray.forEach((entry) => {
      tempCptWeights[entry["label"]] = 1 / cptArray.length;
    });
    setCptWeights(tempCptWeights);
  };

  const sendProcessRequest = async () => {
    if (cptName in cptData) {
      setUploadError(true);
      setUploadErrorText(CONSTANTS.NAME_ERROR);
      setFlashNameUploadError(true);
      await wait(1000);
      setFlashNameUploadError(false);
    } else if (!Utils.errorCheckFloatInput(gwl)) {
      setUploadError(true);
      setUploadErrorText(CONSTANTS.GWL_ERROR);
      setFlashGWLUploadError(true);
      await wait(1000);
      setFlashGWLUploadError(false);
    } else if (!Utils.errorCheckFloatInput(nar)) {
      setUploadError(true);
      setUploadErrorText(CONSTANTS.NAR_ERROR);
      setFlashNARUploadError(true);
      await wait(1000);
      setFlashNARUploadError(false);
    } else {
      setUploadError(false);
      setLoading(true);
      let serverResponse = false;
      const formData = new FormData();
      formData.append(file.name, file);
      formData.append(
        file.name + "_formData",
        JSON.stringify({
          cptName: cptName,
          gwl: gwl,
          nar: nar,
          iskPa: iskPa ? "True" : "False",
        })
      );
      await fetch(CONSTANTS.VS_API_URL + CONSTANTS.CREATE_CPTS_ENDPOINT, {
        method: "POST",
        body: formData,
      })
        .then(async (response) => {
          if (response.ok) {
            serverResponse = true;
            const responseData = await response.json();
            // Set Table Data
            let tempCptTableData = cptTableData;
            if (file.name.split(".")[1] === "xlsx") {
              readXlsxFile(file).then((rows) => {
                rows.shift();
                tempCptTableData[cptName] = rows;
              });
            } else {
              Papa.parse(file, {
                header: true,
                skipEmptyLines: true,
                complete: function (results, file) {
                  tempCptTableData[cptName] = results.data;
                },
              });
            }
            setCptTableData(tempCptTableData);
            // Set CPT Select Dropdown
            let tempOptionArray = cptOptions;
            let tempCPTData = cptData;
            for (const key of Object.keys(responseData)) {
              tempOptionArray.push({
                value: responseData[key],
                label: responseData[key]["name"],
              });
              tempCPTData[key] = responseData[key];
            }
            setCPTData(tempCPTData);
            setCPTOptions(tempOptionArray);
            changeCptWeights(tempOptionArray);
            addToVsProfilePlot(tempCPTData, selectedCorrelations);
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

  const sendCPTMidpointRequest = async (cptsToSend) => {
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.CPT_MIDPOINT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cptsToSend),
    }).then(async (response) => {
      const responseData = await response.json();
      // Add to MidpointData
      let tempMidpointData = cptMidpointData;
      for (const key of Object.keys(responseData)) {
        tempMidpointData[key] = responseData[key];
      }
      setCptMidpointData(tempMidpointData);
    });
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

  const sendVsProfileRequest = async (cptsToSend, correlationsToSend) => {
    const jsonBody = {
      cpts: cptsToSend,
      correlations: correlationsToSend,
    };
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_FROM_CPT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jsonBody),
    }).then(async (response) => {
      const responseData = await response.json();
      // Add to vsProfileData
      let tempVsProfileData = vsProfileData;
      for (const key of Object.keys(responseData)) {
        tempVsProfileData[key] = responseData[key];
      }
      setVsProfileData(tempVsProfileData);
    });
  };

  const sendAverageRequest = async (vsProfilesToSend) => {
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_AVERAGE_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vsProfiles: vsProfilesToSend,
        vsWeights: cptWeights,
        vsCorrelationWeights: correlationWeights,
        vs30CorrelationWeights: {},
      }),
    }).then(async (response) => {
      const responseData = await response.json();
      // Set the Plot Average Data
      setVsProfileAveragePlotData(responseData["average"]);
    });
  };

  const addToVsProfilePlot = async (newCptData, newSelectedCorrelations) => {
    let correlationsToSend = [];
    let cptsToSend = {};
    newSelectedCorrelations.forEach((entry) => {
      for (const cptKey of Object.keys(newCptData)) {
        if (!vsProfileData.hasOwnProperty(cptKey + "_" + entry["label"])) {
          correlationsToSend.push(entry["label"]);
          cptsToSend[cptKey] = newCptData[cptKey];
        }
      }
    });
    if (correlationsToSend.length > 0) {
      await sendVsProfileRequest(cptsToSend, correlationsToSend);
    }
    // Check if new midpoint requests are needed
    let vsProfileToSend = [];
    newSelectedCorrelations.forEach((entry) => {
      for (const cptKey of Object.keys(newCptData)) {
        if (
          !vsProfileMidpointData.hasOwnProperty(cptKey + "_" + entry["label"])
        ) {
          vsProfileToSend.push(vsProfileData[cptKey + "_" + entry["label"]]);
        }
      }
    });
    if (vsProfileToSend.length > 0) {
      await sendVsProfileMidpointRequest(vsProfileToSend);
    }
    // Adds to Plot data from midpoint data
    let tempPlotData = {};
    newSelectedCorrelations.forEach((entry) => {
      for (const cptKey of Object.keys(newCptData)) {
        tempPlotData[cptKey + "_" + entry["label"]] =
          vsProfileMidpointData[cptKey + "_" + entry["label"]];
      }
    });
    setVsProfilePlotData(tempPlotData);
  };

  const changeCPTSelection = async (entries) => {
    // Gather Midpoint data
    let cptsToSend = [];
    let cptLabels = [];
    entries.forEach((entry) => {
      cptLabels.push(entry["label"]);
      if (!cptMidpointData.hasOwnProperty(entry["label"])) {
        cptsToSend.push(entry);
      }
    });
    if (cptsToSend.length !== 0) {
      await sendCPTMidpointRequest(cptsToSend);
    }

    let tempPlotData = {};
    // Create CPT Plot Data
    cptLabels.forEach((name) => {
      tempPlotData[name] = cptMidpointData[name];
    });
    setCptPlotData(tempPlotData);
    setSelectedCptPlot(entries);
  };

  const checkWeights = async () => {
    let checkCor = Utils.errorCheckWeights(correlationWeights);
    let checkCPT = Utils.errorCheckWeights(cptWeights);
    if (!checkCor) {
      setFlashCorWeightError(true);
      setWeightError(true);
    }
    if (!checkCPT) {
      setFlashCPTWeightError(true);
      setWeightError(true);
    }
    if (checkCor && checkCPT) {
      let tempCPTResults = [];
      Object.keys(vsProfileData).forEach(function (key) {
        tempCPTResults.push({ label: key, value: vsProfileData[key] });
      });
      setCptResults(tempCPTResults);
      // Remove average for now
      // sendAverageRequest(vsProfileData);
      // Ensures the values are floats
      Object.keys(correlationWeights).forEach(function (key) {
        correlationWeights[key] = parseFloat(correlationWeights[key]);
      });
      Object.keys(cptWeights).forEach(function (key) {
        cptWeights[key] = parseFloat(cptWeights[key]);
      });
      setCptCorrelationWeights(correlationWeights);
      setWeightError(false);
    } else {
      await wait(1000);
      setFlashCPTWeightError(false);
      setFlashCorWeightError(false);
    }
  };

  // Change the CPT Weights
  const changeCPTWeights = (newWeights) => {
    setCptWeights(newWeights);
  };

  // Change the Correlation Weights
  const changeCorrelationWeights = (newWeights) => {
    setCorrelationWeights(newWeights);
  };

  const onSelectCPT = (e) => {
    setSelectedCptTable(e);
    setSelectedCptTableData(cptTableData[e["label"]]);
    setCptInfo(e["value"]["info"]);
    setDataIsKPa(e["value"]["is_kpa"]);
  };

  const onSelectCorrelations = (e) => {
    setSelectedCorrelations(e);
    addToVsProfilePlot(cptData, e);
  };

  const onSetFile = (e) => {
    setFile(e);
    setCptName(e.name.split(".")[0]);
  };

  const removeFile = (fileToRemove) => {
    let newCptOptions = [];
    let newCptData = [];
    cptOptions.forEach((object) => {
      if (object["label"] !== fileToRemove["label"]) {
        newCptOptions.push(object);
        newCptData[object["label"]] = cptData[object["label"]];
      }
    });
    setCPTData(newCptData);
    setCPTOptions(newCptOptions);
    changeCptWeights(newCptOptions);
    let newVsPlotData = {};
    let newVsProfileData = {};
    let newVsProfileMidpointData = {};
    newCptOptions.forEach((cptOption) => {
      selectedCorrelations.forEach((correlation) => {
        let key = cptOption["label"] + "_" + correlation["label"];
        if (key !== fileToRemove["label"] + "_" + correlation["label"]) {
          newVsPlotData[key] = vsProfilePlotData[key];
          newVsProfileData[key] = vsProfileData[key];
          newVsProfileMidpointData[key] = vsProfileMidpointData[key];
        }
      });
    });
    setVsProfilePlotData(newVsPlotData);
    setVsProfileData(newVsProfileData);
    setVsProfileMidpointData(newVsProfileMidpointData);
    if (
      selectedCptTable !== null &&
      fileToRemove["label"] === selectedCptTable["label"]
    ) {
      setSelectedCptTable(null);
      setSelectedCptTableData(null);
    }
    let cptLabels = [];
    let newSelected = [];
    selectedCptPlot.forEach((entry) => {
      if (entry["label"] !== fileToRemove["label"]) {
        cptLabels.push(entry["label"]);
        newSelected.push(entry);
      }
    });
    let tempPlotData = {};
    // Create CPT Plot Data
    cptLabels.forEach((name) => {
      tempPlotData[name] = cptMidpointData[name];
    });
    setCptPlotData(tempPlotData);
    setSelectedCptPlot(newSelected);
    setCptMidpointData(tempPlotData);
  };

  return (
    <div>
      <div className="row three-column-row cpt-data">
        <div className="col-2 centre-elm cpt-file-manage">
          <div className="process-cpt">
            <div className="form-section-title">Upload CPT</div>
            <div className="outline form-section">
              <div
                className={
                  flashFileUploadError
                    ? "cpt-flash-warning row two-colum-row form-file-input-section"
                    : "row two-colum-row form-file-input-section temp-border"
                }
              >
                <input
                  className="col-8 form-file-input"
                  type="file"
                  onChange={(e) => onSetFile(e.target.files[0])}
                />
                <div className="col-1 file-info">
                  <InfoTooltip text={CONSTANTS.CPT_FILE} />
                </div>
              </div>
              <div className="form-label">CPT Name</div>
              <div className="stretch">
                <input
                  className={
                    flashNameUploadError
                      ? "cpt-flash-warning text-input"
                      : "cpt-input text-input"
                  }
                  value={cptName}
                  onChange={(e) => setCptName(e.target.value)}
                />
              </div>
              <div className="form-label">Ground Water Level (m)</div>
              <div className="stretch">
                <input
                  className={
                    flashGWLUploadError
                      ? "cpt-flash-warning text-input"
                      : "cpt-input text-input"
                  }
                  value={gwl}
                  onChange={(e) => setGWL(e.target.value)}
                />
              </div>
              <div className="form-label">Net Area Ratio</div>
              <div className="stretch">
                <input
                  className={
                    flashNARUploadError
                      ? "cpt-flash-warning text-input"
                      : "cpt-input text-input"
                  }
                  value={nar}
                  onChange={(e) => setNAR(e.target.value)}
                />
              </div>
              <div className="form-label">CPT Units</div>
              <div className="stretch switch-input-section">
                <div className={iskPa ? "normal-label" : "highlight-label"}>
                  MPa
                </div>
                <Switch
                  size="medium"
                  onChange={(e) => setiskPa(e.target.checked)}
                />
                <div className={iskPa ? "highlight-label" : "normal-label"}>
                  kPa
                </div>
              </div>
              <div className="row two-colum-row add-cpt-section temp-border">
                <button
                  disabled={loading}
                  className={
                    flashServerError
                      ? "trans-btn add-cpt-btn form btn btn-danger"
                      : "trans-btn add-cpt-btn form btn btn-primary"
                  }
                  onClick={() => sendProcessRequest()}
                >
                  Add CPT
                </button>
                <div className="col-1 weight-error">
                  {uploadError && (
                    <InfoTooltip text={uploadErrorText} error={true} />
                  )}
                </div>
              </div>
            </div>
          </div>
          <div className="file-section center-elm">
            <div className="form-section-title">CPT Files</div>
            <div className="file-table-section outline form center-elm">
              {Object.keys(cptOptions).length > 0 && (
                <FileTable
                  files={cptOptions}
                  removeFunction={removeFile}
                ></FileTable>
              )}
            </div>
          </div>
        </div>
        <div className="col-4 cpt-table-section">
          <div className="center-elm">
            <div className="form-section-title">CPT Table</div>
            <Select
              className="select-cpt select-box"
              placeholder="Select CPT"
              options={cptOptions}
              isDisabled={cptOptions.length === 0}
              value={selectedCptTable}
              onChange={(e) => onSelectCPT(e)}
            ></Select>
            <div className="outline center-elm cpt-table">
              {Object.keys(cptTableData).length > 0 &&
                selectedCptTable !== null && (
                  <CptTable
                    cptTableData={selectedCptTableData}
                    cptInfo={cptInfo}
                    dataIsKPa={dataIsKPa}
                  ></CptTable>
                )}
            </div>
          </div>
        </div>
        <div className="col-7 cpt-plot-section">
          <div className="center-elm">
            <div className="form-section-title">CPT Plot</div>
            <Select
              className="select-cpt select-box"
              placeholder="Select CPTs"
              isMulti={true}
              options={cptOptions}
              isDisabled={cptOptions.length === 0}
              value={selectedCptPlot}
              onChange={(e) => changeCPTSelection(e)}
            ></Select>
          </div>
          <div className="outline cpt-plot">
            {Object.keys(cptPlotData).length > 0 && (
              <CPTPlot cptPlotData={cptPlotData} gwl={gwl}></CPTPlot>
            )}
          </div>
        </div>
      </div>
      <div className="hr"></div>
      <div className="center-elm">
        <Select
          className="select-cor select-box"
          placeholder="Select CPT - Vs Correlations"
          isMulti={true}
          options={correlationsOptions}
          isDisabled={correlationsOptions.length === 0}
          onChange={(e) => onSelectCorrelations(e)}
        ></Select>
      </div>
      <div className="row two-column-row cor-section">
        <div className="outline col-3 weights center-elm">
          <div className="form-section-title">CPT Weights</div>
          <div className="outline center-elm cpt-weights">
            {Object.keys(cptWeights).length > 0 && (
              <WeightTable
                weights={cptWeights}
                setFunction={changeCPTWeights}
                flashError={flashCPTWeightError}
              />
            )}
          </div>
          <div className="form-section-title">CPT - Vs Correlation Weights</div>
          <div className="outline center-elm cor-weights">
            {Object.keys(correlationWeights).length > 0 && (
              <WeightTable
                weights={correlationWeights}
                setFunction={changeCorrelationWeights}
                flashError={flashCorWeightError}
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
        <div className="col-4 vs-preview-section center-elm">
          <div className="form-section-title">VsProfile Preview</div>
          <div className="outline vs-preview-plot">
            {Object.keys(vsProfilePlotData).length > 0 && (
              <VsProfilePreviewPlot
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

export default memo(CPT);
