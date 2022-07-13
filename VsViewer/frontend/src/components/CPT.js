import React, { useState, useContext, memo } from "react";
import Select from "react-select";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";

import "assets/cpt.css";
import CPTPlot from "components/CptPlot";

const CPT = () => {
  const {
    setCPTData,
    cptMidpointData,
    setCptMidpointData,
  } = useContext(GlobalContext);

  const [filenames, setFilenames] = useState("");
  const [loading, setLoading] = useState(false);
  const [cptOptions, setCPTOptions] = useState([]);
  const [cptPlotData, setCptPlotData] = useState({});

  const sendProcessRequest = async () => {
    setLoading(true);
    const formData = new FormData();
    for (const file of filenames) {formData.append(file.name, file)};
    const requestOptions = {
      method: "POST",
      body: formData,
    }
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
    setLoading(false);
  }

  const sendMidpointRequest = async (cptsToSend) => {
    const requestOptions = {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cptsToSend)
    }
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.MIDPOINT_ENDPOINT, requestOptions)
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
  }

  return (
    <div>
      <div className="process-cpt">
        <div className="form-section-title">Upload CPT files</div>
        <input className="form-file-input" type="file" multiple={true} onChange={(e) => setFilenames(e.target.files)}/>
        <button disabled={loading} className="form btn btn-primary" onClick={() => sendProcessRequest()}>Process CPT's</button>
      </div>
      <div className="hr"></div>
      <div className="center-btn">
        <Select
          className="select-cpt"
          placeholder="Select your CPT's"
          isMulti={true}
          options={cptOptions}
          isDisabled={cptOptions.length === 0}
          onChange={(e) => changeCPTSelection(e)}
        ></Select>
      </div>
      <div className="row three-column-row cpt-data">
        <div className="temp col-3 cpt-table">Table</div>
        <div className="temp col-5 cpt-plot">
          {Object.keys(cptPlotData).length > 0 && (<CPTPlot cptPlotData={cptPlotData}></CPTPlot>)}
        </div>
        <div className="temp col-3 vs-preview-plot">Vs Profile Preview</div>
      </div>
      <div className="hr"></div>
      <div className="center-btn">
        <Select
          className="select-cpt"
          placeholder="Select Correlations"
          isMulti={true}
          options={cptOptions}
          isDisabled={cptOptions.length === 0}
        ></Select>
      </div>
      <div className="row two-column-row weights">
        <div className="temp col-3 cpt-weights">CPT Weights</div>
        <div className="temp col-3 cor-weights">Correlation Weights</div>
      </div>
    </div>
    
  );
};


export default memo(CPT);
