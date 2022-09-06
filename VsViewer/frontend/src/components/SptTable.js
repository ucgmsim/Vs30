import React, { memo } from "react";

import * as Utils from "Utils";

import "assets/sptTable.css";

const SPTTable = ({ sptTableData }) => {
  if (sptTableData !== undefined && sptTableData !== null) {
    const sptTableRows = [];
    const sptInfo = sptTableData["info"];
    sptTableData["depth"].forEach((depth, rowIdx) => {
      sptTableRows.push(
        <tr key={rowIdx}>
          <td className="tbl-col-size">{Utils.roundValue(depth)}</td>
          <td className="tbl-col-size">
            {Utils.roundValue(sptTableData["N"][rowIdx])}
          </td>
          <td className="tbl-col-size">
            {Utils.roundValue(sptTableData["N60"][rowIdx])}
          </td>
        </tr>
      );
    });

    return (
      <div className="center-elm">
        <table className="spt-raw table thead-dark table-striped table-bordered mt-2 w-auto">
          <thead>
            <tr>
              <th className="tbl-col-size" scope="col">
                Depth (m)
              </th>
              <th className="tbl-col-size" scope="col">
                N
              </th>
              <th className="tbl-col-size" scope="col">
                N60 (Computed)
              </th>
            </tr>
          </thead>
          <tbody className="tbl-width spt-scroll-tbl">
            <tr>
              <td className="tbl-width" colSpan="4">
                <div className="spt-scroll-tbl">
                  <table className="spt-tbl-width">
                    <tbody>{sptTableRows}</tbody>
                  </table>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <div className="row two-column-row center-elm spt-info">
          <div className="spt-min-max col-3 center-elm">
            <table className="spt-min-max table thead-dark table-striped table-bordered mt-2 w-auto">
              <tbody>
                <tr>
                  <td className="bold">Min Depth (m)</td>
                  <td className="text-size">{Utils.roundValue(sptInfo["z_min"])}</td>
                </tr>
                <tr>
                  <td className="bold">Max Depth (m)</td>
                  <td className="text-size">{Utils.roundValue(sptInfo["z_max"])}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="spt-removed col-3 center-elm">
            <table className="spt-removed table thead-dark table-striped table-bordered mt-2 w-auto">
              <tbody>
                <tr>
                  <td className="bold rem-label">Depth Spread (m)</td>
                  <td className="text-size">{Utils.roundValue(sptInfo["z_spread"])}</td>
                </tr>
                <tr className="highlight">
                  <td className="bold info-width rem-label">Removed Rows</td>
                  <td className="text-size">{sptInfo["removed_rows"].length}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }
};

export default memo(SPTTable);
