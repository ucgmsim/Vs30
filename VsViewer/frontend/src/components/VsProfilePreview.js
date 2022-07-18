import React, { memo } from "react";

import Plot from "react-plotly.js";

import * as CONSTANTS from "Constants";
import "assets/vsProfilePreview.css";

const VsProfilePreviewPlot = ({ vsProfilePlotData }) => {
  if (Object.keys(vsProfilePlotData).length > 0) {
    let VsArr = [];
    let colourCounter = 0;
    for (const name of Object.keys(vsProfilePlotData)) {
      VsArr.push({
        x: vsProfilePlotData[name]["Vs"],
        y: vsProfilePlotData[name]["Depth"],
        type: "scatter",
        mode: "lines",
        line: { color: CONSTANTS.DEFAULTCOLOURS[colourCounter % 10] },
        name: name,
        hoverinfo: "none",
      });
      // Standard Deviations
      VsArr.push({
        x: vsProfilePlotData[name]["VsSDAbove"],
        y: vsProfilePlotData[name]["Depth"],
        type: "scatter",
        mode: "lines",
        line: {
          color: CONSTANTS.DEFAULTCOLOURS[colourCounter % 10],
          dash: "dash",
        },
        name: name,
        hoverinfo: "none",
      });
      VsArr.push({
        x: vsProfilePlotData[name]["VsSDBelow"],
        y: vsProfilePlotData[name]["Depth"],
        type: "scatter",
        mode: "lines",
        line: {
          color: CONSTANTS.DEFAULTCOLOURS[colourCounter % 10],
          dash: "dash",
        },
        name: name,
        hoverinfo: "none",
      });
      colourCounter += 1;
    }

    return (
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
          },
          autosize: true,
          margin: {
            l: 40,
            r: 40,
            b: 60,
            t: 10,
            pad: 1,
          },
          showlegend: false,
        }}
        useResizeHandler={true}
      />
    );
  }
};

export default memo(VsProfilePreviewPlot);
