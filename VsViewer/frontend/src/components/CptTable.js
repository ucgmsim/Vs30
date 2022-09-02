import React, { memo } from "react";

import * as Utils from "Utils";
import * as CONSTANTS from "Constants";
import { InfoTooltip } from "components";

import "assets/cptTable.css";

const CPTTable = ({ cptTableData, cptInfo }) => {
  if (cptTableData !== undefined && cptTableData !== null) {
    const cptTableRows = [];
    cptTableData.forEach((row, rowIdx) => {
      const rowClassName = "col-size";
      cptTableRows.push(
        <tr
          className={
            cptInfo["Removed rows"].includes(rowIdx) ? "highlight" : ""
          }
          key={rowIdx}
        >
          <td className={rowClassName}>{Utils.roundValue(row["Depth"])}</td>
          <td className={rowClassName}>{Utils.roundValue(row["Qc"])}</td>
          <td className={rowClassName}>{Utils.roundValue(row["Fs"])}</td>
          <td className={rowClassName}>{Utils.roundValue(row["u"])}</td>
        </tr>
      );
    });

    return (
      <div>
        <table className="cpt-raw table thead-dark table-striped table-bordered mt-2 w-auto">
          <thead>
            <tr>
              <th className="col-size" scope="col">
                Depth (m)
              </th>
              <th className="col-size" scope="col">
                qc (MPa)
              </th>
              <th className="col-size" scope="col">
                fs (MPa)
              </th>
              <th className="col-size" scope="col">
                u (MPa)
              </th>
            </tr>
          </thead>
          <tbody className="tbl-width scroll-tbl">
            <tr>
              <td className="tbl-width" colSpan="4">
                <div className="scroll-tbl">
                  <table className="tbl-width">
                    <tbody>{cptTableRows}</tbody>
                  </table>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <div className="row two-column-row center-elm info">
          <div className="min-max col-6 center-elm">
            <table className="table thead-dark table-striped table-bordered mt-2 w-auto">
              <tbody>
                <tr>
                  <td className="bold">Min Depth (m)</td>
                  <td>{Utils.roundValue(cptInfo["z_min"])}</td>
                </tr>
                <tr>
                  <td className="bold">Max Depth (m)</td>
                  <td>{Utils.roundValue(cptInfo["z_max"])}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="removed-info col-6 center-elm">
            <table className=" table thead-dark table-striped table-bordered mt-2 w-auto">
              <tbody>
                <tr>
                  <td className="bold info-width">Depth Spread (m)</td>
                  <td>{Utils.roundValue(cptInfo["z_spread"])}</td>
                </tr>
                <tr className="highlight">
                  <td className="bold info-width">
                  <div className="row two-colum-row info-width">
                    <div className="col-9">
                      Removed Rows
                    </div>
                    <div className=" col-1 file-info-tbl">
                      <InfoTooltip text={CONSTANTS.CPT_REMOVED_ROWS} />
                    </div>
                  </div>
                  </td>
                  <td>{cptInfo["Removed rows"].length}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }
};

export default memo(CPTTable);
