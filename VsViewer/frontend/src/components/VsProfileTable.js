import React, { memo } from "react";

import * as Utils from "Utils";
import * as CONSTANTS from "Constants";
import { InfoTooltip } from "components";

import "assets/vsProfileTable.css";

const VsProfileTable = ({ vsProfileData, vsProfileInfo }) => {
  if (vsProfileData !== undefined && vsProfileData !== null) {
    const vsProfileRows = [];
    vsProfileData.forEach((row, rowIdx) => {
      vsProfileRows.push(
        <tr
          className={
            vsProfileInfo["removed_rows"].includes(rowIdx) ? "highlight" : ""
          }
          key={rowIdx}
        >
          <td className="col-size">{Utils.roundValue(row["Depth"])}</td>
          <td className="col-size">{Utils.roundValue(row["Vs"])}</td>
          <td className="col-size">{Utils.roundValue(row["Vs_SD"])}</td>
        </tr>
      );
    });

    return (
      <div className="center-elm">
        <table className="vs-raw table thead-dark table-striped table-bordered mt-2 w-auto">
          <thead>
            <tr>
              <th className="col-size" scope="col">
                Depth (m)
              </th>
              <th className="col-size" scope="col">
                Vs (m/s)
              </th>
              <th className="col-size" scope="col">
              σ
              </th>
            </tr>
          </thead>
          <tbody className="tbl-width vs-scroll-tbl">
            <tr>
              <td className="tbl-width" colSpan="4">
                <div className="vs-scroll-tbl">
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
                  <td>{Utils.roundValue(vsProfileInfo["z_min"])}</td>
                </tr>
                <tr>
                  <td className="bold">Max Depth (m)</td>
                  <td>{Utils.roundValue(vsProfileInfo["z_max"])}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="vs-removed col-3 center-elm">
            <table className="vs-removed table thead-dark table-striped table-bordered mt-2 w-auto">
              <tbody>
                <tr>
                  <td className="bold">Depth Spread (m)</td>
                  <td>{Utils.roundValue(vsProfileInfo["z_spread"])}</td>
                </tr>
                <tr className="highlight">
                  <td className="bold info-width">
                    <div className="row two-colum-row info-width">
                      <div className="col-9">
                        Removed Rows
                      </div>
                      <div className=" col-1 file-info-tbl">
                        <InfoTooltip text={CONSTANTS.VS_REMOVED_ROWS} />
                      </div>
                    </div>
                  </td>
                  <td>{vsProfileInfo["removed_rows"].length}</td>
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
