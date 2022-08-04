import React, { memo, useState, useEffect } from "react";

import Plot from "react-plotly.js";

import * as CONSTANTS from "Constants";
import "assets/vsProfilePreview.css";

const VsProfilePreviewPlot = ({ vsProfilePlotData, average }) => {
  const [VsArr, setVsArr] = useState([]);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (Object.keys(average).length > 0) {
      updatePlotData(checked);
    }
  }, [average]);

  const updatePlotData = (hideSD) => {
    let tempVsArr = [];
    let colourCounter = 0;
    for (const name of Object.keys(vsProfilePlotData)) {
      tempVsArr.push({
        x: vsProfilePlotData[name]["Vs"],
        y: vsProfilePlotData[name]["Depth"],
        type: "scatter",
        mode: "lines",
        line: { color: CONSTANTS.DEFAULTCOLOURS[colourCounter % 10] },
        name: name,
        hoverinfo: "none",
        legendgroup: name,
      });
      // Standard Deviations
      if (!hideSD) {
        tempVsArr.push({
          x: vsProfilePlotData[name]["VsSDAbove"],
          y: vsProfilePlotData[name]["Depth"],
          type: "scatter",
          mode: "lines",
          line: {
            color: CONSTANTS.DEFAULTCOLOURS[colourCounter % 10],
            dash: "dash",
          },
          hoverinfo: "none",
          showlegend: false,
          legendgroup: name,
        });
        tempVsArr.push({
          x: vsProfilePlotData[name]["VsSDBelow"],
          y: vsProfilePlotData[name]["Depth"],
          type: "scatter",
          mode: "lines",
          line: {
            color: CONSTANTS.DEFAULTCOLOURS[colourCounter % 10],
            dash: "dash",
          },
          hoverinfo: "none",
          showlegend: false,
          legendgroup: name,
        });
      }
      colourCounter += 1;
    }
    if (Object.keys(average).length > 0) {
      tempVsArr.push({
        x: average["Vs"],
        y: average["Depth"],
        type: "scatter",
        mode: "lines",
        line: { color: "black" },
        name: "Average",
        hoverinfo: "none",
        legendgroup: "Average",
      });
      // Standard Deviations
      if (!hideSD) {
        tempVsArr.push({
          x: average["VsSDAbove"],
          y: average["Depth"],
          type: "scatter",
          mode: "lines",
          line: {
            color: "black",
            dash: "dash",
          },
          hoverinfo: "none",
          showlegend: false,
          legendgroup: "Average",
        });
        tempVsArr.push({
          x: average["VsSDBelow"],
          y: average["Depth"],
          type: "scatter",
          mode: "lines",
          line: {
            color: "black",
            dash: "dash",
          },
          hoverinfo: "none",
          showlegend: false,
          legendgroup: "Average",
        });
      }
    }
    setVsArr(tempVsArr);
  };

  const onCheckChange = (newValue) => {
    setChecked(newValue);
    updatePlotData(newValue);
  };

  if (Object.keys(vsProfilePlotData).length > 0) {
    if (VsArr.length === 0) {
      updatePlotData(false);
    }
    return (
      <div className="check-plot-height">
        <label className="hide-sd">
          Hide SD
          <input
            className="hide-sd-check"
            type="checkbox"
            onChange={(e) => onCheckChange(e.target.checked)}
          />
        </label>
        <Plot
          className={"vs-profile-preview-plot"}
          data={VsArr}
          config={{ displayModeBar: false }}
          layout={{
            xaxis: {
              title: "Vs (m/s)",
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
            },
            yaxis: {
              autorange: "reversed",
              title: "Depth (m)",
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
              rangemode: "tozero",
            },
            autosize: true,
            margin: {
              l: 40,
              r: 40,
              b: 60,
              t: 10,
              pad: 1,
            },
            showlegend: true,
            legend: { x: 0, y: -0.15, orientation: "h" },
          }}
          useResizeHandler={true}
        />
      </div>
    );
  }
};

export default memo(VsProfilePreviewPlot);
