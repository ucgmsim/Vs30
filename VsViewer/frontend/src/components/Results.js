import React, { memo, useState, useContext, useEffect } from "react";
import Select from "react-select";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";

import "assets/Results.css";
import * as Utils from "Utils";
import { WeightTable, VsProfilePreviewPlot } from "components";

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
    allCorrelationWeights,
    setAllCorrelationWeights,
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
  const [canSet, setCanSet] = useState(false);

  // Set the Section Weights
  useEffect(() => {
    if (selectedSections.length > 0) {
      let tempSectionWeights = {};
      selectedSections.forEach((entry) => {
        tempSectionWeights[entry["label"]] = 1 / selectedSections.length;
      });
      setSectionWeights(tempSectionWeights);
    }
  }, [selectedSections]);

  // Set the Correlation Weights
  useEffect(() => {
    if (selectedCorrelations.length > 0) {
      let tempCorrelationWeights = {};
      selectedCorrelations.forEach((entry) => {
        tempCorrelationWeights[entry["label"]] =
          1 / selectedCorrelations.length;
      });
      setCorrelationWeights(tempCorrelationWeights);
    }
  }, [selectedCorrelations]);

  // Check the user can set Weights
  useEffect(() => {
    if (selectedCorrelations.length > 0 && selectedSections.length > 0) {
      setCanSet(true);
      if (VsProfileOptions.length > 0) {
        setCanCompute(true);
      }
    } else {
      setCanSet(false);
      setCanCompute(false);
    }
  }, [selectedCorrelations, selectedSections]);

  // Get Sections
  useEffect(() => {
    let tempOptionArray = [];
    let tempVsProfileOptions = [];
    if (Object.keys(vsProfileResults).length > 0) {
      tempOptionArray.push({ value: "VsProfile", label: "VsProfile" });
      for (const key of Object.keys(vsProfileResults)) {
        tempVsProfileOptions.push({ value: vsProfileResults[key], label: key });
      }
    }
    if (Object.keys(cptResults).length > 0) {
      tempOptionArray.push({ value: "CPT", label: "CPT" });
      for (const key of Object.keys(cptResults)) {
        tempVsProfileOptions.push({ value: cptResults[key], label: key });
      }
    }
    if (Object.keys(sptResults).length > 0) {
      tempOptionArray.push({ value: "SPT", label: "SPT" });
      for (const key of Object.keys(sptResults)) {
        tempVsProfileOptions.push({ value: sptResults[key], label: key });
      }
    }
    setSections(tempOptionArray);
    setVsProfileOptions(tempVsProfileOptions);
    if (tempOptionArray.length > 0) {
      setCanCompute(true);
    } else {
      setCanCompute(false);
    }
  }, [vsProfileResults, cptResults, sptResults]);

  // Get Correlations on page load
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

  const computeVs30 = async () => {
    // Get all weights and reweight based on section weights
    let weights = {};
    for (const sectionKey of Object.keys(sectionWeights)) {
      if (sectionKey === "CPT") {
        for (const key of Object.keys(cptWeights)) {
          weights[key] = sectionWeights[sectionKey] * cptWeights[key];
        }
      }
      if (sectionKey === "SPT") {
        for (const key of Object.keys(sptWeights)) {
          weights[key] = sectionWeights[sectionKey] * sptWeights[key];
        }
      }
      if (sectionKey === "VsProfile") {
        for (const key of Object.keys(vsProfileWeights)) {
          weights[key] = sectionWeights[sectionKey] * vsProfileWeights[key];
        }
      }
    }
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_VS30_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vsProfiles: VsProfileOptions,
        vsWeights: weights,
        vsCorrelationWeights: allCorrelationWeights,
        vs30CorrelationWeights: correlationWeights,
      }),
    }).then(async (response) => {
      const responseData = await response.json();
      // Set Vs30 results
      setVs30(responseData["Vs30"]);
      setVs30SD(responseData["Vs30_SD"]);
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

  const checkWeights = () => {
    // TODO error checking
    console.log("Test Weights");
  };

  return (
    <div className="row three-column-row center-elm results-page">
      <div className="col-1">
        <div className="weights-section center-elm">
          <div className="results-title">Section Weights</div>
          <Select
            className="select-box"
            placeholder="Select your Section's"
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
              />
            )}
          </div>
          <div className="vs-weight-title">VsZ - Vs30 Correlation Weights</div>
          <Select
            className="select-box"
            placeholder="Select your Correlation's"
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
                />
              )}
            </div>
            <button
            disabled={!canSet}
            className="preview-btn btn btn-primary"
            onClick={() => checkWeights()}
          >
            Set Weights
          </button>
          </div>

        </div>
      </div>
      <div className="col-5 center-elm result-plot-section">
        <div className="center-elm">
          <div className="vs-plot-title">VsProfile Plot</div>
          <Select
            className="vs-select"
            placeholder="Select your VsProfile's"
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
      <div className="col-1 center-elm vs30-section outline">
        <button
          disabled={!canCompute}
          className="form btn btn-primary compute-btn"
          onClick={() => computeVs30()}
        >
          Compute Vs30
        </button>
        <div className="vs30-title">Vs30</div>
        <div className="vs30-value outline">
          {vs30 === null ? "" : Utils.roundValue(vs30)}
        </div>
        <div className="vs30-title">Vs30 Sigma</div>
        <div className="vs30-value outline">
          {vs30SD === null ? "" : Utils.roundValue(vs30SD)}
        </div>
      </div>
    </div>
  );
};

export default memo(Results);
