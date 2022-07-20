import React, { memo, useState, Select } from "react";

import "assets/spt.css";
import {
  WeightTable,
  VsProfilePreviewPlot,
} from "components";

const SPT = () => {

  // VsProfilePreview
  const [vsProfileData, setVsProfileData] = useState({});
  const [vsProfilePlotData, setVsProfilePlotData] = useState({});
  // Form variables
  const [sptWeights, setSptWeights] = useState({});
  const [correlationWeights, setCorrelationWeights] = useState({});
  const [correlationsOptions, setCorrelationsOptions] = useState([]);
  const [selectedCorrelations, setSelectedCorrelations] = useState([]);
  const [canSet, setCanSet] = useState(false);

  const addToVsProfilePlot = async (selectedCorrelations) => {
    let correlationsToSend = [];
    selectedCorrelations.forEach((entry) => {
      if (!vsProfileData.hasOwnProperty(entry["label"])) {
        correlationsToSend.push(entry["label"]);
      }
    });
    if (correlationsToSend.length > 0) {
      await sendVsProfileRequest(correlationsToSend);
    }
    // Check if new midpoint requests are needed
    let vsProfileToSend = [];
    selectedCorrelations.forEach((entry) => {
      for (const cptKey of Object.keys(cptData)) {
        if (
          !vsProfileMidpointData.hasOwnProperty(cptKey + "_" + entry["label"])
        ) {
          vsProfileToSend.push(vsProfileData[cptKey + "_" + entry["label"]]);
        }
      }
    });
    if (vsProfileToSend.length > 0) {
      await sendVsProfileMidpointRequest(vsProfileToSend);
    }
    // Adds to Plot data from midpoint data
    let tempPlotData = {};
    selectedCorrelations.forEach((entry) => {
      for (const cptKey of Object.keys(cptData)) {
        tempPlotData[cptKey + "_" + entry["label"]] =
          vsProfileMidpointData[cptKey + "_" + entry["label"]];
      }
    });
    setVsProfilePlotData(tempPlotData);
  };

  const onSelectCorrelations = (e) => {
    setSelectedCorrelations(e);
    addToVsProfilePlot(e);
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

  return <div>
    <div className="row two-column-row center-elm spt-top">
      <div className="outline col-3 add-spt">
        Add SPT
      </div>
      <div className="col-7 center-elm spt-plot-section">
        <div className="outline spt-plot">
        SPT Plot
        </div>
      </div>
    </div>
    <div className="hr"/>
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
        <div className="col-7 vs-preview-section-spt center-elm">
          <div className="form-section-title">VsProfile Preview</div>
          <div className="outline vs-preview-plot-spt">
            {Object.keys(vsProfilePlotData).length > 0 && (
              <VsProfilePreviewPlot vsProfilePlotData={vsProfilePlotData} />
            )}
          </div>
        </div>
      </div>
  </div>;
};

export default memo(SPT);
