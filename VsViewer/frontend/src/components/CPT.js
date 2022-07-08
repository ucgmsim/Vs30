import React, { useState, useContext, memo } from "react";
import Select from "react-select";

import { GlobalContext } from "context";
import * as CONSTANTS from "Constants";

import "assets/cpt.css";

const CPT = () => {
  const {
    cptData,
    setCPTData,
  } = useContext(GlobalContext);

  const [filenames, setFilenames] = useState("");
  const [loading, setLoading] = useState(false);
  const [cptOptions, setCPTOptions] = useState([]);
  const [cptNames, setCPTNames] = useState([]);

  const sendRequest = async () => {
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

  return (
    <div>
      <div className="process-cpt">
        <div className="form-section-title">Upload CPT files</div>
        <input className="form-file-input" type="file" multiple={true} onChange={(e) => setFilenames(e.target.files)}/>
        <button disabled={loading} className="form btn btn-primary" onClick={() => sendRequest()}>Process CPT's</button>
      </div>
      <div className="hr"></div>
      <Select
        className="select-cpt"
        placeholder="Select your CPT's"
        options={cptOptions}
        isDisabled={cptOptions.length === 0}
      ></Select>
      <div className="row three-column-row cpt-data">
        <div className="temp col-3 cpt-table">Table</div>
        <div className="temp col-5 cpt-plot">CPT Plot</div>
        <div className="temp col-3 vs-preview-plot">Vs Profile Preview</div>
      </div>
      <div className="hr"></div>
      <Select
        className="select-cpt"
        placeholder="Select Correlations"
        options={cptOptions}
        isDisabled={cptOptions.length === 0}
      ></Select>
      <div className="row two-column-row weights">
        <div className="temp col-3 cpt-weights">CPT Weights</div>
        <div className="temp col-3 cor-weights">Correlation Weights</div>
      </div>
    </div>
    
  );
};


export default memo(CPT);
