import React, { memo, useState, useEffect } from "react";

import Plot from "react-plotly.js";

import * as CONSTANTS from "Constants";
import "assets/sptPlot.css";

const SPTPlot = ({ sptPlotData }) => {
  const [data, setData] = useState([]);

  useEffect(() => {
    updatePlotData();
  }, [sptPlotData]);

  const updatePlotData = () => {
    let tempData = [];
    let colourCounter = 0;
    if (Object.keys(sptPlotData).length > 0) {
      for (const name of Object.keys(sptPlotData)) {
        let NData = {
          x: sptPlotData[name]["N"],
          y: sptPlotData[name]["Depth"],
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
        let N60Data = {
          x: sptPlotData[name]["N60"],
          y: sptPlotData[name]["Depth"],
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
        tempData.push(NData);
        tempData.push(N60Data);
        colourCounter += 1;
      }
      setData(tempData);
    }
  };

  if (Object.keys(sptPlotData).length > 0) {
    if (data.length === 0) {
      updatePlotData();
    }
    return (
      <div className="spt-plots">
        <Plot
          className="spt-chart"
          data={data}
          config={{ displayModeBar: false, responsive: true }}
          title="N"
          layout={{
            rows: 1,
            columns: 2,
            pattern: "independent",
            autosize: true,
            xaxis: { title: "N", domain: [0, 0.475] },
            xaxis2: { title: "N60", domain: [0.525, 1] },
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
                text: "N",
                showarrow: false,
                x: 0.5,
                xref: "x domain",
                y: 1.03,
                yref: "y domain",
              },
              {
                text: "N60",
                showarrow: false,
                x: 0.5,
                xref: "x2 domain",
                y: 1.03,
                yref: "y domain",
              },
            ],
          }}
          useResizeHandler={true}
        ></Plot>
      </div>
    );
  }
};

export default memo(SPTPlot);
