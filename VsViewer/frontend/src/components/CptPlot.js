import React, { memo, useState, useEffect } from "react";

import Plot from "react-plotly.js";

import * as CONSTANTS from "Constants";
import "assets/cptPlot.css";

const CPTPlot = ({ cptPlotData, gwl }) => {
  const [data, setData] = useState([]);

  useEffect(() => {
    updatePlotData();
  }, [cptPlotData]);

  const updatePlotData = () => {
    let tempData = [];
    let colourCounter = 0;
    if (Object.keys(cptPlotData).length > 0) {
      for (const name of Object.keys(cptPlotData)) {
        let QcData = {
          x: cptPlotData[name]["Qc"],
          y: cptPlotData[name]["Depth"],
          xaxis: "x1",
          yaxis: "y1",
          legendgroup: name,
          name: name,
          mode: "lines",
          type: "scatter",
          hoverinfo: "none",
          line: {
            color: CONSTANTS.DEFAULTCOLOURS[colourCounter % 10],
          },
        };
        let FsData = {
          x: cptPlotData[name]["Fs"],
          y: cptPlotData[name]["Depth"],
          xaxis: "x2",
          legendgroup: name,
          name: name,
          title: "N",
          yaxis: "y1",
          mode: "lines",
          type: "scatter",
          hoverinfo: "none",
          showlegend: false,
          line: {
            color: CONSTANTS.DEFAULTCOLOURS[colourCounter % 10],
          },
        };
        let UData = {
          x: cptPlotData[name]["u"],
          y: cptPlotData[name]["Depth"],
          xaxis: "x3",
          legendgroup: name,
          name: name,
          title: "N",
          yaxis: "y1",
          mode: "lines",
          type: "scatter",
          hoverinfo: "none",
          showlegend: false,
          line: {
            color: CONSTANTS.DEFAULTCOLOURS[colourCounter % 10],
          },
        };
        tempData.push(QcData);
        tempData.push(FsData);
        tempData.push(UData);
        colourCounter += 1;
      }
      let GWLLegend = {
        x: [undefined],
        name: "GWL",
        mode: "lines",
        line: {
          color: "rgba(0, 0, 0, 0.60)",
        },
      };
      tempData.push(GWLLegend);
      setData(tempData);
    }
  };

  if (Object.keys(cptPlotData).length > 0) {
    if (data.length === 0) {
      updatePlotData();
    }
    return (
      <Plot
        className="cpt-plots"
        data={data}
        config={{ displayModeBar: false, responsive: true }}
        title="N"
        layout={{
          rows: 1,
          columns: 3,
          pattern: "independent",
          autosize: true,
          xaxis: { title: "qc (MPa)", domain: [0, 0.325] },
          xaxis2: { title: "fs (MPa)", domain: [0.341, 0.658] },
          xaxis3: { title: "u (MPa)", domain: [0.674, 1] },
          yaxis: { autorange: "reversed", title: "Depth (m)" },
          margin: {
            l: 55,
            r: 40,
            b: 60,
            t: 40,
            pad: 1,
          },
          showlegend: true,
          legend: { x: 0, y: -0.15, orientation: "h" },
          annotations: [
            {
              text: "qc",
              showarrow: false,
              x: 0.5,
              xref: "x domain",
              y: 1.03,
              yref: "y domain",
            },
            {
              text: "fs",
              showarrow: false,
              x: 0.5,
              xref: "x2 domain",
              y: 1.03,
              yref: "y domain",
            },
            {
              text: "u",
              showarrow: false,
              x: 0.5,
              xref: "x3 domain",
              y: 1.03,
              yref: "y domain",
            },
          ],
          shapes: [
            {
              type: "line",
              xref: "paper",
              yref: "y",
              x0: 0,
              y0: gwl,
              x1: 1,
              y1: gwl,
              line: {
                color: "rgba(0, 0, 0, 0.60)",
                width: 2.5,
              },
            },
          ],
        }}
        useResizeHandler={true}
      ></Plot>
    );
  }
};

export default memo(CPTPlot);
