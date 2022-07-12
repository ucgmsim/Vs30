import React, { useState, useEffect, useContext, memo } from "react";
import Select from "react-select";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";

import "assets/cpt.css";
import CPTPlot from "components/CptPlot";

const CPT = () => {
  const {
    setCPTData,
    cptPlotData,
    cptMidpointData,
  } = useContext(GlobalContext);

  const [filenames, setFilenames] = useState("");
  const [loading, setLoading] = useState(false);
  const [cptOptions, setCPTOptions] = useState([]);
  const [cptNames, setCPTNames] = useState([]);

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
        // Add to cptMidpointData
        for (const key of Object.keys(responseData)) {
          cptMidpointData[key] = responseData[key];
        }
    });
  }

  useEffect(() => {
    if (cptNames.length !== 0) {
      let cptsToSend = [];
      if (Object.keys(cptMidpointData).length === 0) {
        cptsToSend = cptNames;
      } else {
        cptNames.forEach((cptName) => {
           if (!cptMidpointData.hasOwnProperty(cptName["label"])) {
            cptsToSend.push(cptName);
          }
        });
      }
      if (cptsToSend.length !== 0) {
        console.log("CPT - sending Requst");
        console.log(cptsToSend);
        sendMidpointRequest(cptsToSend);
      }
      console.log("CPT - cptNames");
      console.log(cptNames);
      console.log("CPT - cptPlotData");
      console.log(cptPlotData);
    }
  }, [cptNames]);

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
          onChange={(e) => setCPTNames(e)}
        ></Select>
      </div>
      <div className="row three-column-row cpt-data">
        <div className="temp col-3 cpt-table">Table</div>
        <div className="temp col-5 cpt-plot">
          {cptNames.length !== 0 && (<CPTPlot cptNames={cptNames}></CPTPlot>)}
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
