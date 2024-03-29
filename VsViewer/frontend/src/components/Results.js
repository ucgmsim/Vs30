import React, { memo, useState, useContext, useEffect } from "react";
import Select from "react-select";
import JSZip from "jszip";
import { saveAs } from "file-saver";
import { wait } from "@testing-library/user-event/dist/utils";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";

import "assets/Results.css";
import * as Utils from "Utils";
import { WeightTable, VsProfilePreviewPlot, InfoTooltip } from "components";

const Results = () => {
  const {
    vsProfileMidpointData,
    setVsProfileMidpointData,
    vsProfileWeights,
    cptWeights,
    sptWeights,
    vsProfileResults,
    cptResults,
    sptResults,
    cptCorrelationWeights,
    sptCorrelationWeights,
  } = useContext(GlobalContext);

  // Weights
  const [sections, setSections] = useState([]);
  const [selectedSections, setSelectedSections] = useState([]);
  const [sectionWeights, setSectionWeights] = useState({});
  const [correlationOptions, setCorrelationOptions] = useState([]);
  const [correlationWeights, setCorrelationWeights] = useState({});
  const [selectedCorrelations, setSelectedCorrelations] = useState([]);
  // VsProfile Plot
  const [selectedVsProfilePlot, setSelectedVsProfilePlot] = useState(null);
  const [vsProfilePlotData, setVsProfilePlotData] = useState({});
  const [vsProfileAveragePlotData, setVsProfileAveragePlotData] = useState({});
  // Form variables
  const [VsProfileOptions, setVsProfileOptions] = useState([]);
  const [vs30, setVs30] = useState(null);
  const [vs30SD, setVs30SD] = useState(null);
  const [canCompute, setCanCompute] = useState(false);
  const [canSetWeights, setCanSetWeights] = useState(false);
  // Errors
  const [flashCorWeightError, setFlashCorWeightError] = useState(false);
  const [flashSecWeightError, setFlashSecWeightError] = useState(false);
  const [weightError, setWeightError] = useState(false);
  const [flashComputeError, setFlashComputeError] = useState(false);
  const [computeError, setComputeError] = useState(false);
  const [computeErrorText, setComputeErrorText] = useState(
    CONSTANTS.COMPUTE_ERROR
  );

  // Set the Section Weights
  useEffect(() => {
    let tempSectionWeights = {};
    selectedSections.forEach((entry) => {
      tempSectionWeights[entry["label"]] = 1 / selectedSections.length;
    });
    setSectionWeights(tempSectionWeights);
    if (selectedSections.length === 0) {
      setCanCompute(false);
    }
  }, [selectedSections]);

  // Set the Correlation Weights
  useEffect(() => {
    let tempCorrelationWeights = {};
    selectedCorrelations.forEach((entry) => {
      tempCorrelationWeights[entry["label"]] = 1 / selectedCorrelations.length;
    });
    setCorrelationWeights(tempCorrelationWeights);
    if (selectedCorrelations.length === 0) {
      setCanCompute(false);
    }
  }, [selectedCorrelations]);

  // Check the user can set Weights
  useEffect(() => {
    if (selectedCorrelations.length > 0 && selectedSections.length > 0) {
      setCanSetWeights(true);
    } else {
      setCanSetWeights(false);
    }
  }, [selectedCorrelations, selectedSections]);

  // Get Sections
  useEffect(() => {
    let tempOptionArray = [];
    let tempVsProfileOptions = [];
    if (cptResults.length > 0) {
      tempOptionArray.push({ value: "CPT", label: "CPT" });
      cptResults.forEach((entry) => {
        tempVsProfileOptions.push(entry);
      });
    }
    if (sptResults.length > 0) {
      tempOptionArray.push({ value: "SPT", label: "SPT" });
      sptResults.forEach((entry) => {
        tempVsProfileOptions.push(entry);
      });
    }
    if (vsProfileResults.length > 0) {
      tempOptionArray.push({ value: "VsProfile", label: "VsProfile" });
      vsProfileResults.forEach((entry) => {
        tempVsProfileOptions.push(entry);
      });
    }
    setSections(tempOptionArray);
    setVsProfileOptions(tempVsProfileOptions);
  }, [vsProfileResults, cptResults, sptResults]);

  // Get Correlations on page load
  useEffect(() => {
    if (correlationOptions.length === 0) {
      fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_CORRELATIONS_ENDPOINT, {
        method: "GET",
      }).then(async (response) => {
        const responseData = await response.json();
        // Set Correlation Select Dropdown
        let tempOptionArray = [];
        for (const value of Object.values(responseData)) {
          tempOptionArray.push({ value: value, label: value });
        }
        setCorrelationOptions(tempOptionArray);
      });
    }
  }, []);

  const computeVs30 = async () => {
    setVs30(null);
    setVs30SD(null);
    // Get all weights and reweight based on section weights
    let serverResponse = false;
    let weights = {};
    let vsProfilesToSend = {};
    for (const sectionKey of Object.keys(sectionWeights)) {
      if (sectionKey === "CPT") {
        for (const key of Object.keys(cptWeights)) {
          weights[key] = sectionWeights[sectionKey] * cptWeights[key];
          VsProfileOptions.forEach((entry) => {
            if (entry["label"].includes(key)) {
              vsProfilesToSend[entry["label"]] = entry["value"];
            }
          });
        }
      }
      if (sectionKey === "SPT") {
        for (const key of Object.keys(sptWeights)) {
          weights[key] = sectionWeights[sectionKey] * sptWeights[key];
          VsProfileOptions.forEach((entry) => {
            if (entry["label"].includes(key)) {
              vsProfilesToSend[entry["label"]] = entry["value"];
            }
          });
        }
      }
      if (sectionKey === "VsProfile") {
        for (const key of Object.keys(vsProfileWeights)) {
          weights[key] = sectionWeights[sectionKey] * vsProfileWeights[key];
          VsProfileOptions.forEach((entry) => {
            if (entry["label"].includes(key)) {
              vsProfilesToSend[entry["label"]] = entry["value"];
            }
          });
        }
      }
    }
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_VS30_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vsProfiles: vsProfilesToSend,
        vsWeights: weights,
        cptVsCorrelationWeights: cptCorrelationWeights,
        sptVsCorrelationWeights: sptCorrelationWeights,
        vs30CorrelationWeights: correlationWeights,
      }),
    })
      .then(async (response) => {
        if (response.ok) {
          serverResponse = true;
          const responseData = await response.json();
          // Set Vs30 results
          setVs30(responseData["Vs30"]);
          setVs30SD(responseData["Vs30_SD"]);
          setComputeError(false);
        } else {
          setComputeErrorText(CONSTANTS.COMPUTE_ERROR);
          setComputeError(true);
          setFlashComputeError(true);
          await wait(1000);
          setFlashComputeError(false);
        }
      })
      .catch(async () => {
        if (serverResponse) {
          setComputeErrorText(CONSTANTS.COMPUTE_ERROR);
        } else {
          setComputeErrorText(CONSTANTS.REQUEST_ERROR);
        }
        setComputeError(true);
        setFlashComputeError(true);
        await wait(1000);
        setFlashComputeError(false);
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

  const downloadData = () => {
    const createRows = (vsProfile) => {
      let rows = [["Depth", "Vs", "Vs_SD"]];
      for (let i = 0; i < vsProfile.depth.length; i++) {
        rows.push([
          `${vsProfile.depth[i]}`,
          `${vsProfile.vs[i]}`,
          `${vsProfile.vs_sd[i]}`,
        ]);
      }
      return rows.map((e) => e.join(",")).join("\n");
    };
    var zip = new JSZip();

    VsProfileOptions.forEach((entry) => {
      zip.file(entry["label"] + ".csv", createRows(entry["value"]));
    });

    zip.generateAsync({ type: "blob" }).then(function (content) {
      saveAs(content, "VsProfile Data.zip");
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

  // Change the section Weights
  const changeSectionWeights = (newWeights) => {
    setSectionWeights(newWeights);
  };

  // Change the Correlation Weights
  const changeCorrelationWeights = (newWeights) => {
    setCorrelationWeights(newWeights);
  };

  const checkWeights = async () => {
    let checkCor = Utils.errorCheckWeights(correlationWeights);
    let checkSections = Utils.errorCheckWeights(sectionWeights);
    if (!checkCor) {
      setFlashCorWeightError(true);
      setWeightError(true);
    }
    if (!checkSections) {
      setFlashSecWeightError(true);
      setWeightError(true);
    }
    if (checkCor && checkSections) {
      setCanCompute(true);
      setWeightError(false);
      // Ensures the values are floats
      Object.keys(correlationWeights).forEach(function (key) {
        correlationWeights[key] = parseFloat(correlationWeights[key]);
      });
      Object.keys(sectionWeights).forEach(function (key) {
        sectionWeights[key] = parseFloat(sectionWeights[key]);
      });
    } else {
      setCanCompute(false);
      await wait(1000);
      setFlashSecWeightError(false);
      setFlashCorWeightError(false);
    }
  };

  return (
    <div className="row three-column-row results-page">
      <div className="col-1 weights-section center-elm outline">
        <div className="results-title">Section Weights</div>
        <Select
          className="small-select-box"
          placeholder="Select Sections"
          options={sections}
          isMulti={true}
          isDisabled={sections.length === 0}
          value={selectedSections}
          onChange={(e) => setSelectedSections(e)}
        ></Select>
        <div className="outline center-elm result-weights">
          {Object.keys(sectionWeights).length > 0 && (
            <WeightTable
              weights={sectionWeights}
              setFunction={changeSectionWeights}
              flashError={flashSecWeightError}
            />
          )}
        </div>
        <div className="vs-weight-title">VsZ - Vs30 Correlation Weights</div>
        <Select
          className="small-select-box"
          placeholder="Select VsZ - Vs30 Correlations"
          options={correlationOptions}
          isMulti={true}
          isDisabled={correlationOptions.length === 0}
          value={selectedCorrelations}
          onChange={(e) => setSelectedCorrelations(e)}
        ></Select>
        <div className="outline center-elm result-weights">
          <div className="result-weight-area">
            {Object.keys(correlationWeights).length > 0 && (
              <WeightTable
                weights={correlationWeights}
                setFunction={changeCorrelationWeights}
                flashError={flashCorWeightError}
              />
            )}
          </div>
        </div>
        <div className="row two-colum-row set-weights-section">
          <button
            disabled={!canSetWeights}
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
      <div className="col-5 center-elm result-plot-section">
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
        <div className="outline results-vs-plot">
          {Object.keys(vsProfilePlotData).length > 0 && (
            <VsProfilePreviewPlot
              className="vs-plot"
              vsProfilePlotData={vsProfilePlotData}
              average={vsProfileAveragePlotData}
            />
          )}
        </div>
      </div>
      <div className="col-1 center-elm vs30-results-section">
        <div className="center-elm vs30-section outline">
          <div className="row two-colum-row compute-section temp-border">
            <button
              disabled={!canCompute}
              className={
                flashComputeError
                  ? "trans-btn form btn btn-danger compute-btn"
                  : "trans-btn form compute-btn btn btn-primary"
              }
              onClick={() => computeVs30()}
            >
              Compute Vs30
            </button>
            <div className="col-1 weight-error">
              {computeError && (
                <InfoTooltip text={computeErrorText} error={true} />
              )}
            </div>
          </div>
          <div className="vs30-title">Vs30 (m/s)</div>
          <div className="vs30-value outline">
            {vs30 === null ? "" : Utils.roundValue(vs30)}
          </div>
          <div className="vs30-title">Vs30 Sigma</div>
          <div className="vs30-value outline">
            {vs30SD === null ? "" : Utils.roundValue(vs30SD)}
          </div>
        </div>
        <div className="results-download-section outline">
          <button
            disabled={!canCompute}
            className="form btn btn-primary download-btn"
            onClick={() => downloadData()}
          >
            Download Data
          </button>
        </div>
      </div>
    </div>
  );
};

export default memo(Results);
