import React, { memo } from "react";

import Plot from "react-plotly.js";

import "assets/sptPlot.css";

const SPTPlot = ({ sptPlotData }) => {
  if (Object.keys(sptPlotData).length > 0) {
    let NArr = [];
    let N60Arr = [];
    for (const name of Object.keys(sptPlotData)) {
      NArr.push({
        x: sptPlotData[name]["N"],
        y: sptPlotData[name]["Depth"],
        type: "scatter",
        mode: "lines",
        name: name,
        hoverinfo: "none",
      });
      N60Arr.push({
        x: sptPlotData[name]["N60"],
        y: sptPlotData[name]["Depth"],
        type: "scatter",
        mode: "lines",
        name: name,
        hoverinfo: "none",
      });
    }

    return (
      <div className="row two-column-row spt-plots">
        <Plot
          className={"col-4 single-plot"}
          data={NArr}
          config={{ displayModeBar: false }}
          layout={{
            xaxis: {
              title: "N",
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
            title: "Qc",
            titlefont: {
              size: 22,
            },
            autosize: true,
            margin: {
              l: 40,
              r: 40,
              b: 60,
              t: 40,
              pad: 1,
            },
            showlegend: false,
          }}
          useResizeHandler={true}
        />
        <Plot
          className={"col-4 single-plot"}
          data={N60Arr}
          config={{ displayModeBar: false }}
          layout={{
            xaxis: {
              title: "N60",
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
            title: "Fs",
            titlefont: {
              size: 22,
            },
            autosize: true,
            margin: {
              l: 40,
              r: 40,
              b: 60,
              t: 40,
              pad: 1,
            },
            showlegend: false,
          }}
          useResizeHandler={true}
        />
      </div>
    );
  }
};

export default memo(SPTPlot);
