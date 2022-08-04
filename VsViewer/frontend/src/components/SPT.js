import React, { memo, useState, useContext, useEffect } from "react";
import Select from "react-select";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";

import "assets/spt.css";
import {
  WeightTable,
  VsProfilePreviewPlot,
  SPTPlot,
  SptTable,
} from "components";

const SPT = () => {
  const {
    sptData,
    setSPTData,
    sptMidpointData,
    setSptMidpointData,
    vsProfileMidpointData,
    setVsProfileMidpointData,
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
  // Form variables
  const [file, setFile] = useState("");
  const [boreholeDiameter, setBoreholeDiameter] = useState(150);
  const [energyRatio, setEnergyRatio] = useState("");
  const [sptOptions, setSPTOptions] = useState([]);
  const [sptWeights, setSptWeights] = useState({});
  const [correlationWeights, setCorrelationWeights] = useState({});
  const [correlationsOptions, setCorrelationsOptions] = useState([]);
  const [selectedCorrelations, setSelectedCorrelations] = useState([]);
  const [hammerTypeOptions, setHammerTypeOptions] = useState([]);
  const [hammerType, setHammerType] = useState(null);
  const [soilTypeOptions, setSoilTypeOptions] = useState([]);
  const [soilType, setSoilType] = useState(null);
  const [loading, setLoading] = useState(false);
  const [canSet, setCanSet] = useState(false);

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
    if (selectedCorrelations.length > 0) {
      let tempCorWeights = {};
      selectedCorrelations.forEach((entry) => {
        tempCorWeights[entry["label"]] = 1 / selectedCorrelations.length;
      });
      setCorrelationWeights(tempCorWeights);
    }
  }, [selectedCorrelations]);

  // Check the user can set Weights
  useEffect(() => {
    if (selectedCorrelations.length > 0 && sptOptions.length > 0) {
      setCanSet(true);
    } else {
      setCanSet(false);
    }
  }, [selectedCorrelations, sptOptions]);

  // Get HammerTypes on page load
  if (hammerTypeOptions.length == 0) {
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
  if (soilTypeOptions.length == 0) {
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
  if (correlationsOptions.length == 0) {
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

  const sendProcessRequest = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append(file.name, file);
    formData.append(
      file.name + "_formData",
      JSON.stringify({
        boreholeDiameter: boreholeDiameter,
        energyRatio: energyRatio,
        hammerType: hammerType === null ? "" : hammerType["value"],
        soilType: soilType === null ? "" : soilType["value"],
      })
    );
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.SPT_CREATE_ENDPOINT, {
      method: "POST",
      body: formData,
    }).then(async (response) => {
      const responseData = await response.json();
      // Add to SPT Select Dropdown and SPT Data
      let tempOptions = [];
      let tempSPTData = sptData;
      for (const sptOption of sptOptions) {
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
        tempSPTData[key] = responseData[key];
      }
      setSPTOptions(tempOptions);
      setSPTData(tempSPTData);
    });
    setLoading(false);
    addToVsProfilePlot(selectedCorrelations);
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

  const sendVsProfileRequest = async (correlationsToSend) => {
    const jsonBody = {
      spts: sptData,
      correlations: correlationsToSend,
    };
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VSPROFILE_FROM_SPT_ENDPOINT, {
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

  const addToVsProfilePlot = async (selectedCorrelations) => {
    let correlationsToSend = [];
    selectedCorrelations.forEach((entry) => {
      for (const sptKey of Object.keys(sptData)) {
        if (
          !vsProfileMidpointData.hasOwnProperty(sptKey + "_" + entry["label"])
        ) {
          correlationsToSend.push(entry["label"]);
        }
      }
    });
    if (correlationsToSend.length > 0) {
      await sendVsProfileRequest(correlationsToSend);
    }
    // Check if new midpoint requests are needed
    let vsProfileToSend = [];
    selectedCorrelations.forEach((entry) => {
      for (const sptKey of Object.keys(sptData)) {
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
    let tempPlotData = {};
    selectedCorrelations.forEach((entry) => {
      for (const sptKey of Object.keys(sptData)) {
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
    addToVsProfilePlot(e);
  };

  const onSelectSptTable = (e) => {
    setSelectedSptTable(e);
    setSelectedSptTableData(sptData[e["label"]]);
  };

  const checkWeights = () => {
    console.log("Checking Weights");
    // Will add functionality when Results page is started
  };

  // Change the SPT Weights
  const changeSPTWeights = (newWeights) => {
    setSptWeights(newWeights);
  };

  // Change the Correlation Weights
  const changeCorrelationWeights = (newWeights) => {
    setCorrelationWeights(newWeights);
  };

  return (
    <div>
      <div className="row three-column-row center-elm spt-top">
        <div className="outline col-3 add-spt center-elm">
          <div className="form-section-title">Upload SPT file</div>
          <input
            className="spt-file-input"
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
          />
          <div className="form-label">Borehole Diameter</div>
          <input
            className="text-input"
            value={boreholeDiameter}
            onChange={(e) => setBoreholeDiameter(e.target.value)}
          />
          <div className="form-label">Energy Ratio</div>
          <input
            className="text-input"
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
            isDisabled={soilTypeOptions.length === 0}
            onChange={(e) => setSoilType(e)}
          />
          <button
            disabled={loading}
            className="form btn btn-primary add-spt-btn"
            onClick={() => sendProcessRequest()}
          >
            Add SPT
          </button>
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
          className="select-cor"
          placeholder="Select Correlations"
          isMulti={true}
          options={correlationsOptions}
          isDisabled={correlationsOptions.length === 0}
          onChange={(e) => onSelectCorrelations(e)}
        ></Select>
      </div>
      <div className="row two-column-row center-elm cor-section">
        <div className="outline col-3 weights-spt">
          <div className="form-section-title">SPT Weights</div>
          <div className="outline center-elm spt-weights">
            {Object.keys(sptWeights).length > 0 && (
              <WeightTable
                weights={sptWeights}
                setFunction={changeSPTWeights}
              />
            )}
          </div>
          <div className="form-section-title">Correlation Weights</div>
          <div className="outline center-elm cor-weights-spt">
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
        <div className="col-4 vs-preview-section-spt center-elm">
          <div className="form-section-title">VsProfile Preview</div>
          <div className="outline vs-preview-plot-spt">
            {Object.keys(vsProfilePlotData).length > 0 && (
              <VsProfilePreviewPlot vsProfilePlotData={vsProfilePlotData} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(SPT);
