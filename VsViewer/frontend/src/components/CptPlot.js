import React, { memo } from "react";

import Plot from "react-plotly.js";

import "assets/cptPlot.css";

const CPTPlot = ({ cptPlotData }) => {
  if (Object.keys(cptPlotData).length > 0) {
    let QcArr = [];
    let FsArr = [];
    let uArr = [];
    for (const name of Object.keys(cptPlotData)) {
      QcArr.push({
        x: cptPlotData[name]["Qc"],
        y: cptPlotData[name]["Depth"],
        type: "scatter",
        mode: "lines",
        name: name,
        hoverinfo: "none",
      });
      FsArr.push({
        x: cptPlotData[name]["Fs"],
        y: cptPlotData[name]["Depth"],
        type: "scatter",
        mode: "lines",
        name: name,
        hoverinfo: "none",
      });
      uArr.push({
        x: cptPlotData[name]["u"],
        y: cptPlotData[name]["Depth"],
        type: "scatter",
        mode: "lines",
        name: name,
        hoverinfo: "none",
      });
    }

    return (
      <div className="row three-column-row cpt-plots">
        <Plot
          className={"col-4 single-plot"}
          data={QcArr}
          config={{ displayModeBar: false }}
          layout={{
            xaxis: {
              title: "Qc (MPa)",
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
          data={FsArr}
          config={{ displayModeBar: false }}
          layout={{
            xaxis: {
              title: "Fs (MPa)",
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
        <Plot
          className={"col-4 single-plot"}
          data={uArr}
          config={{ displayModeBar: false }}
          layout={{
            xaxis: {
              title: "u (MPa)",
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
            title: "u",
            titlefont: {
              size: 22,
            },
            autosize: true,
            margin: {
              l: 40,
              r: 0,
              b: 60,
              t: 40,
              pad: 1,
            },
            showlegend: true,
            legend: { x: 0.7, y: 1.1 },
          }}
          useResizeHandler={true}
        />
      </div>
    );
  }
};

export default memo(CPTPlot);
