import React, { useState } from "react";

import * as CONSTANTS from "Constants";
import "assets/cpt.css";

const CPT = () => {
  const [filenames, setFilenames] = useState("");
  const [loading, setLoading] = useState(false);

  const sendRequest = async () => {
    setLoading(true);
    const formData = new FormData();
    for (const file of filenames) {formData.append(file.name, file)};
    const requestOptions = {
      method: "POST",
      body: formData,
    }
    await fetch(CONSTANTS.VS_API_URL + CONSTANTS.CREATE_CPTS_ENDPOINT, requestOptions);
    setLoading(false);
  }

  return (
    <div className="process-cpt">
      <div>
        <div className="form form-section-title">Upload CPT files</div>
        <input className="form" type="file" multiple={true} onChange={(e) => setFilenames(e.target.files)}/>
        <button disabled={loading} className="form btn btn-primary" onClick={() => sendRequest()}>Process CPT's</button>
      </div>
    </div>
  );
};


export default CPT;
