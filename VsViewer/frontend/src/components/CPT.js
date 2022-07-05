import React, { useState } from "react";
import { useFilePicker } from "use-file-picker";

import * as CONSTANTS from "../Constants";

const CPT = () => {
  const [filename, setFilename] = useState("");

  const sendRequest = async () => {
    const formData = new FormData();
    // filesContent.forEach(e => console.log(e));
    // console.log(filename);
    formData.append("file", filename);
    // formData.append("fileName", filename.name);
    // filesContent.forEach(e => formData.append(e));
    // const formData2 = {"file": filename};
    
    const requestOptions = {
      method: "POST",
      body: JSON.stringify({"file": 123}),
      // headers: {
      //   "content-type": "multipart/form-data"
      // },
    }
    console.log(requestOptions);
    return await fetch(CONSTANTS.VS_API_URL + CONSTANTS.CREATE_CPTS_ENDPOINT, requestOptions)
  }

  const [openFileSelector, { filesContent }] = useFilePicker({
    accept: ".csv"
  });

  return (
    <div className="container-fluid max-width">
      <button onClick={() => openFileSelector()}>Upload CPT files</button>
      <input type="file" onChange={(e) => setFilename(e.target.files[0])}/>
      <br />
      {filesContent.map((file, index) => (
        <div key={index}>
          {file.name}
        </div>
      ))}
      <br />
      <button onClick={() => sendRequest()}>Process CPT's</button>
    </div>
  );
};


export default CPT;
