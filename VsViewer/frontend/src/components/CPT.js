import React, { useEffect, useState, useContext, memo } from "react";
import Select from "react-select";
import Papa from "papaparse";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";

import "assets/cpt.css";
import CPTPlot from "components/CptPlot";
import CptTable from "components/CptTable";
import WeightTable from "components/WeightTable";
import VsProfilePreviewPlot from "components/VsProfilePreview";

const CPT = () => {
  const {
    cptData,
    setCPTData,
    cptMidpointData,
    setCptMidpointData,
    vsProfileMidpointData,
    setVsProfileMidpointData
  } = useContext(GlobalContext);

  const [filenames, setFilenames] = useState("");
  const [loading, setLoading] = useState(false);
  const [cptOptions, setCPTOptions] = useState([]);
  const [selectedCptPlot, setSelectedCptPlot] = useState(null);
  const [cptPlotData, setCptPlotData] = useState({});
  const [cptTableData, setCptTableData] = useState({});
  const [selectedCptTable, setSelectedCptTable] = useState(null);
  const [selectedCptTableData, setSelectedCptTableData] = useState(null);
  const [cptInfo, setCptInfo] = useState(null);
  const [correlationsOptions, setCorrelationsOptions] = useState([]);
  const [correlations, setCorrelations] = useState([]);
  const [cptWeights, setCptWeights] = useState({});
  const [correlationWeights, setCorrelationWeights] = useState({});
  const [canCompute, setCanCompute] = useState(false);
  const [vsProfileData, setVsProfileData] = useState({});
  const [vsProfilePlotData, setVsProfilePlotData] = useState({});

  // Need to remove eventually
  const [cptWeightsRefresh, setCptWeightsRefresh] = useState(false);
  const [correlationWeightsRefresh, setCorrelationWeightsRefresh] = useState(false);

  if (correlationsOptions.length == 0) {
    const requestOptions = {
      method: "GET",
    }
    fetch(CONSTANTS.VS_API_URL + CONSTANTS.GET_CORRELATIONS_ENDPOINT, requestOptions)
      .then(async (response) => {
        const responseData = await response.json();
        // Set Correlation Select Dropdown
        let tempOptionArray = [];
        for (const value of Object.values(responseData)) {
          tempOptionArray.push({value:value, label:value});
        }
        setCorrelationsOptions(tempOptionArray);
    });
  }

  const sendProcessRequest = async () => {
    setLoading(true);
    const formData = new FormData();
    for (const file of filenames) {formData.append(file.name, file)};
    const requestOptions = {
      method: "POST",
      body: formData,
    }
    // Send Request
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.CREATE_CPTS_ENDPOINT, requestOptions)
      .then(async (response) => {
        const responseData = await response.json();
        setCPTData(responseData);
        // Set CPT Select Dropdown
        let tempOptionArray = [];
        for (const key of Object.keys(responseData)) {
          tempOptionArray.push({value:responseData[key], label:responseData[key]["name"]});
        }

        setCPTOptions(tempOptionArray);
    });
    // Set Table Data
    let tempCptTableData = {}
    for (const file of filenames) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: function (results, file) {
          tempCptTableData[file.name.split(".")[0]] = results.data
        },
      });
    }
    setCptTableData(tempCptTableData);
    setSelectedCptTable(null);
    setSelectedCptPlot(null);
    setCptPlotData({});
    setLoading(false);
  }

  const sendMidpointRequest = async (cptsToSend) => {
    const requestOptions = {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cptsToSend)
    }
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.CPT_MIDPOINT_ENDPOINT, requestOptions)
      .then(async (response) => {
        const responseData = await response.json();
        // Add to MidpointData
        let tempMidpointData = cptMidpointData;
        for (const key of Object.keys(responseData)) {
          tempMidpointData[key] = responseData[key];
        }
        setCptMidpointData(tempMidpointData);
    });
  }

  const sendVsProfileMidpointRequest = async (vsProfileToSend) => {
    const requestOptions = {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(vsProfileToSend)
    }
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VS_PROFILE_MIDPOINT_ENDPOINT, requestOptions)
      .then(async (response) => {
        const responseData = await response.json();
        // Add to MidpointData
        let tempMidpointData = vsProfileMidpointData;
        for (const key of Object.keys(responseData)) {
          tempMidpointData[key] = responseData[key];
        }
        setVsProfileMidpointData(tempMidpointData);
    });
  }

  const sendVsProfileRequest = async (correlationsToSend) => {
    const jsonBody = {
      "cpts": cptData,
      "correlations": correlationsToSend
    }
    const requestOptions = {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(jsonBody)
    }
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.VSPROFILE_FROM_CPT_ENDPOINT, requestOptions)
      .then(async (response) => {
        const responseData = await response.json();
        // Add to vsProfileData
        let tempVsProfileData = vsProfileData;
        for (const key of Object.keys(responseData)) {
          tempVsProfileData[key] = responseData[key];
        }
        setVsProfileData(tempVsProfileData);
    });
  }

  const addToVsProfilePlot = async (correlations) => {
    let correlationsToSend = [];
    correlations.forEach((entry) => {
      if (!vsProfileData.hasOwnProperty(entry["label"])) {
        correlationsToSend.push(entry["label"]);
      }
    });
    if (correlationsToSend.length > 0) {
      await sendVsProfileRequest(correlationsToSend);
    }
    // Check if new midpoint requests are needed
    let vsProfileToSend = [];
    correlations.forEach((entry) => {
      for (const cptKey of Object.keys(cptData)) {
        if (!vsProfileMidpointData.hasOwnProperty(cptKey + "_" + entry["label"])) {
          vsProfileToSend.push(vsProfileData[cptKey + "_" + entry["label"]]);
        }
      }
    });
    if (vsProfileToSend.length > 0) {
      await sendVsProfileMidpointRequest(vsProfileToSend);
    }
    // Adds to Plot data from midpoint data
    let tempPlotData = {};
    correlations.forEach((entry) => {
      for (const cptKey of Object.keys(cptData)) {
        tempPlotData[cptKey + "_" + entry["label"]] = vsProfileMidpointData[cptKey + "_" + entry["label"]];
      }
    });
    setVsProfilePlotData(tempPlotData);
  }

  const changeCPTSelection = async (entries) => {
    // Gather Midpoint data
    let cptsToSend = [];
    let cptLabels = [];
    entries.forEach((entry) => {
      cptLabels.push(entry["label"]);
      if (!cptMidpointData.hasOwnProperty(entry["label"])) {
        cptsToSend.push(entry);
      }
    })
    if (cptsToSend.length !== 0) {
      await sendMidpointRequest(cptsToSend);
    }

    let tempPlotData = {};
    // Create CPT Plot Data
    cptLabels.forEach((name) => {
      tempPlotData[name] = cptMidpointData[name]
    });
    setCptPlotData(tempPlotData);
    setSelectedCptPlot(entries);
  }

  const checkWeights = () => {
    console.log("Checking Weights");
  }

  // Set the CPT Weights
  useEffect(() => {
    if (cptOptions.length > 0) {
      let tempCptWeights = {};
      cptOptions.forEach((entry) => {
        tempCptWeights[entry["label"]] = 1 / cptOptions.length;
      });
      setCptWeights(tempCptWeights);
    }
  }, [cptOptions]);

  // Set the Correlation Weights
  useEffect(() => {
    if (correlations.length > 0) {
      let tempCorWeights = {};
      correlations.forEach((entry) => {
        tempCorWeights[entry["label"]] = 1 / correlations.length;
      });
      setCorrelationWeights(tempCorWeights);
      setCorrelationWeightsRefresh(!correlationWeightsRefresh);
    }
  }, [correlations]);

  // Check the user can set Weights
  useEffect(() => {
    if (correlations.length > 0 && cptOptions.length > 0) {
      setCanCompute(true);
    } else {
      setCanCompute(false);
    }
  }, [correlations, cptOptions]);

  // Change the CPT Weights
  const changeCPTWeights = (newWeights) => {
    setCptWeights(newWeights);
  }

  // Change the Correlation Weights
  const changeCorrelationWeights = (newWeights) => {
    setCorrelationWeights(newWeights);
  }

  return (
    <div>
      <div className="process-cpt">
        <div className="form-section-title">Upload CPT files</div>
        <input className="form-file-input" type="file" multiple={true} onChange={(e) => setFilenames(e.target.files)}/>
        <button disabled={loading} className="form btn btn-primary" onClick={() => sendProcessRequest()}>Process CPT's</button>
      </div>
      <div className="hr"></div>
      <div className="row two-column-row center-elm cpt-data">
        <div className="col-4 cpt-table">
          <div className="center-elm">
            <div className="form-section-title">CPT Table</div>
            <Select
              className="select-cpt"
              placeholder="Select your CPT's"
              options={cptOptions}
              isDisabled={cptOptions.length === 0}
              value={selectedCptTable}
              onChange={(e) => {
                setSelectedCptTable(e);
                setSelectedCptTableData(cptTableData[e["label"]]);
                setCptInfo(cptData[e["label"]]["info"]);
              }}
            ></Select>
            <div className="outline center-elm cpt-table">
              {Object.keys(cptTableData).length > 0 && selectedCptTable !== null && (<CptTable cptTableData={selectedCptTableData} cptInfo={cptInfo}></CptTable>)}
            </div>
          </div>
        </div>
        <div className="col-7 cpt-plot">
          <div className="center-elm">
            <div className="form-section-title">CPT Plot</div>
            <Select
              className="select-cpt"
              placeholder="Select your CPT's"
              isMulti={true}
              options={cptOptions}
              isDisabled={cptOptions.length === 0}
              value={selectedCptPlot}
              onChange={(e) => changeCPTSelection(e)}
            ></Select>
          </div>
          <div className="outline cpt-plot">
            {Object.keys(cptPlotData).length > 0 && (<CPTPlot cptPlotData={cptPlotData}></CPTPlot>)}
          </div>
        </div>
        
      </div>
      <div className="hr"></div>
      <div className="center-elm">
        <Select
          className="select-cor"
          placeholder="Select Correlations"
          isMulti={true}
          options={correlationsOptions}
          isDisabled={correlationsOptions.length === 0}
          onChange={(e) => {
            setCorrelations(e);
            addToVsProfilePlot(e);
          }}
        ></Select>
      </div>
      <div className="row two-column-row center-elm cor-section">
        <div className="outline col-3 weights">
          <div className="form-section-title">CPT Weights</div>
          <div className="outline center-elm cpt-weights">
            {Object.keys(cptWeights).length > 0 && (<WeightTable weights={cptWeights} setFunction={changeCPTWeights} refresh={cptWeightsRefresh} />)}
          </div>
          <div className="form-section-title">Correlation Weights</div>
          <div className="outline center-elm cor-weights">
            {Object.keys(correlationWeights).length > 0 && (<WeightTable weights={correlationWeights} setFunction={changeCorrelationWeights} refresh={correlationWeightsRefresh} />)}
          </div>
          <button disabled={!canCompute} className="preview-btn btn btn-primary" onClick={() => checkWeights()}>Set Weights</button>
        </div>
        <div className="col-4 vs-preview-section center-elm">
          <div className="form-section-title">VsProfile Preview</div>
          <div className="outline vs-preview-plot">
            {Object.keys(vsProfilePlotData).length > 0 && (<VsProfilePreviewPlot vsProfilePlotData={vsProfilePlotData} />)}
          </div>
        </div>
      </div>
    </div>
    
  );
};


export default memo(CPT);
