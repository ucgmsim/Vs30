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
        console.log(responseData);
        setCPTData(responseData);
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
      <div>
        <Select
          placeholder="Select your CPT's"
          options={cptData.map((option, index) => {option.name})}
          isDisabled={cptData.length === 0}
        ></Select>
      </div>
    </div>
    
  );
};


export default memo(CPT);
