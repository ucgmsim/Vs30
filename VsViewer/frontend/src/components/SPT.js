import React, { memo, useState, useContext, useEffect } from "react";
import Select from "react-select";
import Papa from "papaparse";
import { wait } from "@testing-library/user-event/dist/utils";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";
import * as Utils from "Utils";

import "assets/spt.css";
import {
  WeightTable,
  FileTable,
  VsProfilePreviewPlot,
  SPTPlot,
  SptTable,
  InfoTooltip,
} from "components";

const SPT = () => {
  const {
    sptData,
    setSPTData,
    sptMidpointData,
    setSptMidpointData,
    vsProfileMidpointData,
    setVsProfileMidpointData,
    sptWeights,
    setSptWeights,
    setSptResults,
    setSptCorrelationWeights,
    sptCorrelationWeights,
  } = useContext(GlobalContext);

  // SPT Plot
  const [selectedSptPlot, setSelectedSptPlot] = useState(null);
  const [sptPlotData, setSptPlotData] = useState({});
  // SPT Table
  const [selectedSptTable, setSelectedSptTable] = useState(null);
  const [selectedSptTableData, setSelectedSptTableData] = useState(null);
  // VsProfilePreview
  const [vsProfileData, setVsProfileData] = useState({});
  const [vsProfilePlotData, setVsProfilePlotData] = useState({});
  const [vsProfileAveragePlotData, setVsProfileAveragePlotData] = useState({});
  // Form variables
  const [file, setFile] = useState("");
  const [sptName, setSptName] = useState("");
  const [boreholeDiameter, setBoreholeDiameter] = useState(150);
  const [energyRatio, setEnergyRatio] = useState("");
  const [sptOptions, setSPTOptions] = useState([]);
  const [correlationWeights, setCorrelationWeights] = useState({});
  const [correlationsOptions, setCorrelationsOptions] = useState([]);
  const [selectedCorrelations, setSelectedCorrelations] = useState([]);
  const [hammerTypeOptions, setHammerTypeOptions] = useState([]);
  const [hammerType, setHammerType] = useState(null);
  const [soilTypeOptions, setSoilTypeOptions] = useState([]);
  const [soilType, setSoilType] = useState(null);
  const [userSelectSoil, setUserSelectSoil] = useState(false);
  const [loading, setLoading] = useState(false);
  const [canSet, setCanSet] = useState(false);
  const [flashSPTWeightError, setFlashSPTWeightError] = useState(false);
  const [flashCorWeightError, setFlashCorWeightError] = useState(false);
  const [weightError, setWeightError] = useState(false);
  const [flashFileUploadError, setFlashFileUploadError] = useState(false);
  const [flashNameUploadError, setFlashNameUploadError] = useState(false);
  const [flashBoreUploadError, setFlashBoreUploadError] = useState(false);
  const [flashEnergyUploadError, setFlashEnergyUploadError] = useState(false);
  const [flashServerError, setFlashServerError] = useState(false);
  const [uploadError, setUploadError] = useState(false);
  const [uploadErrorText, setUploadErrorText] = useState(CONSTANTS.FILE_ERROR);

  // Set the SPT Weights
  useEffect(() => {
    if (sptOptions.length > 0) {
      let tempSptWeights = {};
      sptOptions.forEach((entry) => {
        tempSptWeights[entry["label"]] = 1 / sptOptions.length;
      });
      setSptWeights(tempSptWeights);
    }
  }, [sptOptions]);

  // Set the Correlation Weights
  useEffect(() => {
    let tempCorWeights = {};
    let tempNewVsData = {};
    selectedCorrelations.forEach((entry) => {
      tempCorWeights[entry["label"]] = 1 / selectedCorrelations.length;
      sptOptions.forEach((object) => {
        let key = object["label"] + "_" + entry["label"];
        tempNewVsData[key] = vsProfileData[key];
      });
    });
    setVsProfileData(tempNewVsData);
    setCorrelationWeights(tempCorWeights);
  }, [selectedCorrelations]);

  // Check the user can set Weights
  useEffect(() => {
    if (selectedCorrelations.length > 0 && sptOptions.length > 0) {
      setCanSet(true);
    } else {
      setCanSet(false);
    }
  }, [selectedCorrelations, sptOptions]);

  useEffect(() => {
    // Get HammerTypes on page load
    if (hammerTypeOptions.length === 0) {
      fetch(CONSTANTS.VS_API_URL + CONSTANTS.GET_HAMMER_TYPES_ENDPOINT, {
        method: "GET",
      }).then(async (response) => {
        const responseData = await response.json();
        // Set HammerType Select Dropdown
        let tempOptionArray = [];
        for (const value of Object.values(responseData)) {
          tempOptionArray.push({ value: value, label: value });
        }
        setHammerTypeOptions(tempOptionArray);
      });
    }
    // Get SoilTypes on page load
    if (soilTypeOptions.length === 0) {
      fetch(CONSTANTS.VS_API_URL + CONSTANTS.GET_SOIL_TYPES_ENDPOINT, {
        method: "GET",
      }).then(async (response) => {
        const responseData = await response.json();
        // Set SoilTypes Select Dropdown
        let tempOptionArray = [];
        for (const value of Object.values(responseData)) {
          tempOptionArray.push({ value: value, label: value });
        }
        setSoilTypeOptions(tempOptionArray);
      });
    }
    // Get Correlations on page load
    if (correlationsOptions.length === 0) {
      fetch(CONSTANTS.VS_API_URL + CONSTANTS.GET_SPT_CORRELATIONS_ENDPOINT, {
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

  const sendProcessRequest = async () => {
    if (sptName in sptData) {
      setUploadError(true);
      setUploadErrorText(CONSTANTS.NAME_ERROR);
      setFlashNameUploadError(true);
      await wait(1000);
      setFlashNameUploadError(false);
    } else if (!Utils.errorCheckFloatInput(boreholeDiameter)){
      setUploadError(true);
      setUploadErrorText(CONSTANTS.BORE_ERROR);
      setFlashBoreUploadError(true);
      await wait(1000);
      setFlashBoreUploadError(false);
    } else if (energyRatio !== "" && !Utils.errorCheckFloatInput(energyRatio)){
      setUploadError(true);
      setUploadErrorText(CONSTANTS.ENERGY_ERROR);
      setFlashEnergyUploadError(true);
      await wait(1000);
      setFlashEnergyUploadError(false);
    } else {
      setUploadError(false);
      setLoading(true);
      let serverResponse = false;
      const formData = new FormData();
      formData.append(file.name, file);
      formData.append(
        file.name + "_formData",
        JSON.stringify({
          sptName: sptName,
          boreholeDiameter: boreholeDiameter,
          energyRatio: energyRatio,
          hammerType: hammerType === null ? "" : hammerType["value"],
          soilType: soilType === null ? "" : soilType["value"],
        })
      );
      await fetch(CONSTANTS.VS_API_URL + CONSTANTS.SPT_CREATE_ENDPOINT, {
        method: "POST",
        body: formData,
      })
        .then(async (response) => {
          if (response.ok) {
            serverResponse = true;
            const responseData = await response.json();
            // Add to SPT Select Dropdown and SPT Data
            let tempOptions = [];
            let tempSPTData = {};
            for (const sptOption of sptOptions) {
              tempOptions.push({
                value: sptOption["value"],
                label: sptOption["label"],
              });
              tempSPTData[sptOption["label"]] = sptData[sptOption["label"]];
            }
            for (const key of Object.keys(responseData)) {
              tempOptions.push({
                value: responseData[key],
                label: responseData[key]["name"],
              });
              tempSPTData[key] = responseData[key];
            }
            setSPTOptions(tempOptions);
            setSPTData(tempSPTData);
            addToVsProfilePlot(tempSPTData, selectedCorrelations);
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

  const sendVsProfileRequest = async (sptsToSend, correlationsToSend) => {
    const jsonBody = {
      spts: sptsToSend,
      correlations: correlationsToSend,
    };
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_FROM_SPT_ENDPOINT, {
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

  const addToVsProfilePlot = async (newSptData, selectedCorrelations) => {
    let correlationsToSend = [];
    let sptsToSend = {};
    selectedCorrelations.forEach((entry) => {
      for (const sptKey of Object.keys(newSptData)) {
        if (
          !vsProfileMidpointData.hasOwnProperty(sptKey + "_" + entry["label"])
        ) {
          if (!correlationsToSend.includes(entry["label"])) {
            correlationsToSend.push(entry["label"]);
          }
          sptsToSend[sptKey] = newSptData[sptKey];
        }
      }
    });
    if (correlationsToSend.length > 0) {
      await sendVsProfileRequest(sptsToSend, correlationsToSend);
    }
    // Check if new midpoint requests are needed
    let vsProfileToSend = [];
    selectedCorrelations.forEach((entry) => {
      for (const sptKey of Object.keys(newSptData)) {
        if (
          !vsProfileMidpointData.hasOwnProperty(sptKey + "_" + entry["label"])
        ) {
          vsProfileToSend.push(vsProfileData[sptKey + "_" + entry["label"]]);
        }
      }
    });
    if (vsProfileToSend.length > 0) {
      await sendVsProfileMidpointRequest(vsProfileToSend);
    }
    // Adds to Plot data from midpoint data
    let tempPlotData = [];
    selectedCorrelations.forEach((entry) => {
      for (const sptKey of Object.keys(newSptData)) {
        tempPlotData[sptKey + "_" + entry["label"]] =
          vsProfileMidpointData[sptKey + "_" + entry["label"]];
      }
    });
    setVsProfilePlotData(tempPlotData);
  };

  const sendSPTMidpointRequest = async (sptsToSend) => {
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.SPT_MIDPOINT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sptsToSend),
    }).then(async (response) => {
      const responseData = await response.json();
      // Add to MidpointData
      let tempMidpointData = sptMidpointData;
      for (const key of Object.keys(responseData)) {
        tempMidpointData[key] = responseData[key];
      }
      setSptMidpointData(tempMidpointData);
    });
  };

  const sendAverageRequest = async (vsProfilesToSend) => {
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_AVERAGE_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vsProfiles: vsProfilesToSend,
        vsWeights: sptWeights,
        vsCorrelationWeights: correlationWeights,
        vs30CorrelationWeights: {},
      }),
    }).then(async (response) => {
      const responseData = await response.json();
      // Set the Plot Average Data
      setVsProfileAveragePlotData(responseData["average"]);
    });
  };

  const changeSPTSelection = async (entries) => {
    // Gather Midpoint data
    let sptsToSend = [];
    let sptLabels = [];
    entries.forEach((entry) => {
      sptLabels.push(entry["label"]);
      if (!sptMidpointData.hasOwnProperty(entry["label"])) {
        sptsToSend.push(entry);
      }
    });
    if (sptsToSend.length !== 0) {
      await sendSPTMidpointRequest(sptsToSend);
    }
    let tempPlotData = {};
    // Create SPT Plot Data
    sptLabels.forEach((name) => {
      tempPlotData[name] = sptMidpointData[name];
    });
    setSptPlotData(tempPlotData);
    setSelectedSptPlot(entries);
  };

  const onSelectCorrelations = (e) => {
    setSelectedCorrelations(e);
    addToVsProfilePlot(sptData, e);
  };

  const onSelectSptTable = (e) => {
    setSelectedSptTable(e);
    setSelectedSptTableData(sptData[e["label"]]);
  };

  const checkWeights = async () => {
    let checkCor = Utils.errorCheckWeights(correlationWeights);
    let checkSPT = Utils.errorCheckWeights(sptWeights);
    if (!checkCor) {
      setFlashCorWeightError(true);
      setWeightError(true);
    }
    if (!checkSPT) {
      setFlashSPTWeightError(true);
      setWeightError(true);
    }
    if (checkCor && checkSPT) {
      setSptResults(vsProfileData);
      // Remove average for now
      // sendAverageRequest(vsProfileData);
      // Ensures the values are floats
      Object.keys(correlationWeights).forEach(function (key) {
        correlationWeights[key] = parseFloat(correlationWeights[key]);
      });
      Object.keys(sptWeights).forEach(function (key) {
        sptWeights[key] = parseFloat(sptWeights[key]);
      });
      setSptCorrelationWeights(correlationWeights);
      setWeightError(false);
    } else {
      await wait(1000);
      setFlashSPTWeightError(false);
      setFlashCorWeightError(false);
    }
  };

  // Change the SPT Weights
  const changeSPTWeights = (newWeights) => {
    setSptWeights(newWeights);
  };

  // Change the Correlation Weights
  const changeCorrelationWeights = (newWeights) => {
    setCorrelationWeights(newWeights);
  };

  const removeFile = (fileToRemove) => {
    let newSptOptions = [];
    let newSptData = {};
    let newPlotSelected = [];
    let newPlotData = {};
    sptOptions.forEach((object) => {
      if (object["label"] !== fileToRemove["label"]) {
        newSptOptions.push(object);
        newSptData[object["label"]] = sptData[object["label"]];
        newPlotSelected.push(object);
        if (sptPlotData[object["label"]] !== undefined) {
          newPlotData[object["label"]] = sptPlotData[object["label"]];
        }
      }
    });
    setSelectedSptPlot(newPlotSelected);
    setSptPlotData(newPlotData);
    setSPTData(newSptData);
    setSPTOptions(newSptOptions);
    changeSPTWeights(newSptOptions);
    let newVsPlotData = {};
    let newVsProfileData = {};
    let newVsProfileMidpointData = {};
    for (const key of Object.keys(vsProfilePlotData)) {
      selectedCorrelations.forEach((correlation) => {
        if (key !== fileToRemove["label"] + "_" + correlation["label"]) {
          newVsPlotData[key] = vsProfilePlotData[key];
          newVsProfileData[key] = vsProfileData[key];
          newVsProfileMidpointData[key] = vsProfileMidpointData[key];
        }
      });
    }
    setVsProfilePlotData(newVsPlotData);
    setVsProfileData(newVsProfileData);
    setVsProfileMidpointData(newVsProfileMidpointData);
    if (selectedSptTable === fileToRemove) {
      setSelectedSptTable(null);
    }
    setVsProfileAveragePlotData({});
  };

  // Set the file and check for Soil type
  const checkFile = (file) => {
    setFile(file);
    setSptName(file.name.split(".")[0]);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: function (results) {
        if (!results.meta.fields.includes("Soil")) {
          setUserSelectSoil(true);
        } else {
          setUserSelectSoil(false);
        }
      },
    });
  };

  return (
    <div>
      <div className="row three-column-row center-elm spt-top">
        <div className="col-3 upload-section">
          <div className="center-elm spt-form-section">
            <div className="form-section-title">Upload SPT</div>
            <div className="outline add-spt center-elm">
              <div
                className={
                  flashFileUploadError
                    ? "cpt-flash-warning row two-colum-row form-file-input-section"
                    : "row two-colum-row form-file-input-section temp-border"
                }
              >
                <input
                  className="col-8 spt-file-input"
                  type="file"
                  onChange={(e) => checkFile(e.target.files[0])}
                />
                <div className="col-1 file-info">
                  <InfoTooltip text={CONSTANTS.SPT_FILE} />
                </div>
              </div>
              <div className="form-label">SPT Name</div>
              <div className="stretch">
                <input
                  className={
                    flashNameUploadError
                      ? "cpt-flash-warning text-input"
                      : "cpt-input text-input"
                  }
                  value={sptName}
                  onChange={(e) => setSptName(e.target.value)}
                />
              </div>
              <div className="form-label">Borehole Diameter (mm)</div>
              <input
                className={
                  flashBoreUploadError
                    ? "cpt-flash-warning text-input"
                    : "cpt-input text-input"
                }
                value={boreholeDiameter}
                onChange={(e) => setBoreholeDiameter(e.target.value)}
              />
              <div className="form-label">Energy Ratio</div>
              <input
                className={
                  flashEnergyUploadError
                    ? "cpt-flash-warning text-input"
                    : "cpt-input text-input"
                }
                value={energyRatio}
                onChange={(e) => setEnergyRatio(e.target.value)}
              />
              <div className="form-label">Hammer Type</div>
              <Select
                className="spt-select"
                placeholder="Select Hammer Type"
                options={hammerTypeOptions}
                isDisabled={hammerTypeOptions.length === 0}
                onChange={(e) => setHammerType(e)}
              />
              <div className="form-label">Soil Type</div>
              <Select
                className="spt-select"
                placeholder="Select Soil Type"
                options={soilTypeOptions}
                isDisabled={soilTypeOptions.length === 0 || !userSelectSoil}
                onChange={(e) => setSoilType(e)}
              />

              <div
                className={
                  flashServerError
                    ? "cpt-flash-warning row two-colum-row add-spt-section"
                    : "row two-colum-row add-spt-section temp-border"
                }
              >
                <button
                  disabled={loading}
                  className="form btn btn-primary add-spt-btn"
                  onClick={() => sendProcessRequest()}
                >
                  Add SPT
                </button>
                <div className="col-1 weight-error">
                  {uploadError && (
                    <InfoTooltip text={uploadErrorText} error={true} />
                  )}
                </div>
              </div>
            </div>
          </div>
          <div className="spt-file-section center-elm">
            <div className="form-section-title">SPT Files</div>
            <div className="file-table-section outline form center-elm">
              {Object.keys(sptOptions).length > 0 && (
                <FileTable
                  files={sptOptions}
                  removeFunction={removeFile}
                ></FileTable>
              )}
            </div>
          </div>
        </div>

        <div className="col-2 center-elm spt-table-section">
          <div className="center-elm">
            <div className="spt-table-title">SPT Table</div>
            <Select
              className="select-box"
              placeholder="Select your SPT's"
              options={sptOptions}
              isDisabled={sptOptions.length === 0}
              value={selectedSptTable}
              onChange={(e) => onSelectSptTable(e)}
            ></Select>
          </div>
          <div className="outline spt-table">
            {Object.keys(sptData).length > 0 && selectedSptTable !== null && (
              <SptTable sptTableData={selectedSptTableData} />
            )}
          </div>
        </div>
        <div className="col-4 center-elm spt-plot-section">
          <div className="center-elm">
            <div className="spt-plot-title">SPT Plot</div>
            <Select
              className="select-box"
              placeholder="Select your SPT's"
              isMulti={true}
              options={sptOptions}
              isDisabled={sptOptions.length === 0}
              value={selectedSptPlot}
              onChange={(e) => changeSPTSelection(e)}
            ></Select>
          </div>
          <div className="outline spt-plot">
            {Object.keys(sptPlotData).length > 0 && (
              <SPTPlot sptPlotData={sptPlotData} />
            )}
          </div>
        </div>
      </div>
      <div className="hr" />
      <div className="center-elm">
        <Select
          className="select-box"
          placeholder="Select Correlations"
          isMulti={true}
          options={correlationsOptions}
          isDisabled={correlationsOptions.length === 0}
          onChange={(e) => onSelectCorrelations(e)}
        ></Select>
      </div>
      <div className="row two-column-row center-elm cor-section">
        <div className="outline col-3 weights-spt center-elm">
          <div className="form-section-title">SPT Weights</div>
          <div className="outline center-elm spt-weights">
            {Object.keys(sptWeights).length > 0 && (
              <WeightTable
                weights={sptWeights}
                setFunction={changeSPTWeights}
                flashError={flashSPTWeightError}
              />
            )}
          </div>
          <div className="form-section-title">SPT - Vs Correlation Weights</div>
          <div className="outline center-elm cor-weights-spt">
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
        <div className="col-4 vs-preview-section-spt center-elm">
          <div className="form-section-title">VsProfile Preview</div>
          <div className="outline vs-preview-plot-spt">
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

export default memo(SPT);
