import React, { memo } from "react";

import * as Utils from "Utils";
import * as CONSTANTS from "Constants";
import { InfoTooltip } from "components";

import "assets/vsProfileTable.css";

const VsProfileTable = ({ vsProfileData, vsProfileInfo }) => {
  if (vsProfileData !== undefined && vsProfileData !== null) {
    const vsProfileRows = [];
    let rowLabels = Object.keys(vsProfileData[0]);
    vsProfileData.forEach((row, rowIdx) => {
      vsProfileRows.push(
        <tr
          className={
            vsProfileInfo["removed_rows"].includes(rowIdx) ? "highlight" : ""
          }
          key={rowIdx}
        >
          <td className="vs-col-size">{Utils.roundValue(row[rowLabels[0]])}</td>
          <td className="vs-col-size">{Utils.roundValue(row[rowLabels[1]])}</td>
          <td className="vs-col-size">{Utils.roundValue(row[rowLabels[2]])}</td>
        </tr>
      );
    });

    return (
      <div className="center-elm">
        <table className="vs-raw table thead-dark table-striped table-bordered mt-2 w-auto">
          <thead>
            <tr>
              <th className="vs-col-size" scope="col">
                Depth (m)
              </th>
              <th className="vs-col-size" scope="col">
                Vs (m/s)
              </th>
              <th className="vs-col-size" scope="col">
                σ
              </th>
            </tr>
          </thead>
          <tbody className="tbl-width vs-scroll-tbl">
            <tr>
              <td className="tbl-width" colSpan="4">
                <div className="vs-scroll-tbl add-overlap-scrollbar">
                  <table className="vs-tbl-width">
                    <tbody>{vsProfileRows}</tbody>
                  </table>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <div className="row two-column-row center-elm vs-info">
          <div className="vs-min-max col-3 center-elm">
            <table className="vs-min-max table thead-dark table-striped table-bordered mt-2 w-auto">
              <tbody>
                <tr>
                  <td className="bold">Min Depth (m)</td>
                  <td className="text-size">
                    {Utils.roundValue(vsProfileInfo["z_min"])}
                  </td>
                </tr>
                <tr>
                  <td className="bold">Max Depth (m)</td>
                  <td className="text-size">
                    {Utils.roundValue(vsProfileInfo["z_max"])}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="vs-removed col-3 center-elm">
            <table className="vs-removed table thead-dark table-striped table-bordered mt-2 w-auto">
              <tbody>
                <tr>
                  <td className="bold">Depth Spread (m)</td>
                  <td className="text-size">
                    {Utils.roundValue(vsProfileInfo["z_spread"])}
                  </td>
                </tr>
                <tr className="highlight">
                  <td className="bold info-width">
                    <div className="row two-colum-row info-width">
                      <div className="col-9 rem-label">Removed Rows</div>
                      <div className="col-1 file-info-tbl">
                        <InfoTooltip text={CONSTANTS.VS_REMOVED_ROWS} />
                      </div>
                    </div>
                  </td>
                  <td className="text-size">
                    {vsProfileInfo["removed_rows"].length}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }
};

export default memo(VsProfileTable);
