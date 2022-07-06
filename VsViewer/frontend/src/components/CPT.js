import React, { Fragment, useState } from "react";

import * as CONSTANTS from "../Constants";
import "../assets/cpt.css";

const CPT = () => {
  const [filenames, setFilenames] = useState("");

  const sendRequest = async () => {
    const formData = new FormData();
    for (const file of filenames) {formData.append(file.name, file)};
    const requestOptions = {
      method: "POST",
      body: formData,
    }
    return await fetch(CONSTANTS.VS_API_URL + CONSTANTS.CREATE_CPTS_ENDPOINT, requestOptions)
  }

  return (
    <div className="box">
      <div>
        <div className="form form-section-title">Upload CPT files</div>
        <input className="form" type="file" multiple={true} onChange={(e) => setFilenames(e.target.files)}/>
        <button className="form btn btn-primary" onClick={() => sendRequest()}>Process CPT's</button>
      </div>
    </div>
  );
};


export default CPT;
