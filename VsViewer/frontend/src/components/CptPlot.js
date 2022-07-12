import React, { useContext, useEffect } from "react";

import Plot from "react-plotly.js";

import { GlobalContext } from "context";
import "assets/cptPlot.css";

const CPTPlot = (cptNames) => {
  const {
    cptPlotData,
  } = useContext(GlobalContext);

  let QcArr = [];
  let FsArr = [];
  let uArr = [];

  useEffect(() => {
    if (cptNames.length !== 0) {

      QcArr = [];
      for (const key of Object.keys(cptPlotData)) {
        QcArr.push({
          x: cptPlotData[key]["Qc"],
          y: cptPlotData[key]["Depth"],
          type: "scatter",
          mode: "lines",
          name: key,
          hoverinfo: "none",
        });
      }
  
      FsArr = [];
      for (const key of Object.keys(cptPlotData)) {
        FsArr.push({
          x: cptPlotData[key]["Fs"],
          y: cptPlotData[key]["Depth"],
          type: "scatter",
          mode: "lines",
          name: key,
          hoverinfo: "none",
        });
      }
  
      uArr = [];
      for (const key of Object.keys(cptPlotData)) {
        uArr.push({
          x: cptPlotData[key]["u"],
          y: cptPlotData[key]["Depth"],
          type: "scatter",
          mode: "lines",
          name: key,
          hoverinfo: "none",
        });
      }

      console.log("PLOT - cptNames");
      console.log(cptNames);
      console.log("PLOT - cptPlotData");
      console.log(cptPlotData);
      console.log(QcArr);
    }
  }, [cptNames]);

  if (cptNames.length !== 0) {
    console.log("Plot names");
    console.log(cptNames);

    return (
      <div className="row three-column-row cpt-plots">
        <Plot
          className={"col-3 cpt-plot"}
          data={QcArr}
          config={
            {displayModeBar: false}
          }
          layout={{
            xaxis: {
              title: 'Qc (MPa)',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
            },
            yaxis: {
              autorange: 'reversed',
              title: 'Depth (m)',
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
            autosize: false,
            width: 300,
            height: 600,
            margin: {
              l: 40,
              r: 40,
              b: 40,
              t: 40,
              pad: 1
            },
            showlegend: false,
          }}
          useResizeHandler={true}
        />
        <Plot
          className={"col-3 cpt-plot"}
          data={FsArr}
          config={
            {displayModeBar: false}
          }
          layout={{
            xaxis: {
              title: 'Fs (MPa)',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
            },
            yaxis: {
              autorange: 'reversed',
              title: 'Depth (m)',
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
            autosize: false,
            width: 300,
            height: 600,
            margin: {
              l: 40,
              r: 40,
              b: 40,
              t: 40,
              pad: 1
            },
            showlegend: false,
          }}
          useResizeHandler={true}
        />
        <Plot
          className={"col-4 cpt-plot"}
          data={uArr}
          config={
            {displayModeBar: false}
          }
          layout={{
            xaxis: {
              title: 'u (MPa)',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
            },
            yaxis: {
              autorange: 'reversed',
              title: 'Depth (m)',
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
            autosize: false,
            width: 300,
            height: 600,
            margin: {
              l: 40,
              r: 0,
              b: 40,
              t: 40,
              pad: 1
            },
          }}
          useResizeHandler={true}
        />
      </div>
      
    );
  }
};

export default CPTPlot;
