import React, { memo } from "react";

import * as Utils from "Utils";
import * as CONSTANTS from "Constants";
import { InfoTooltip } from "components";

import "assets/cptTable.css";

const CPTTable = ({ cptTableData, cptInfo, dataIsKPa }) => {
  if (cptTableData !== undefined && cptTableData !== null) {
    const cptTableRows = [];
    let rowLabels = Object.keys(cptTableData[0]);

    const rowClassName = "col-size";
    cptTableData.forEach((row, rowIdx) => {
      cptTableRows.push(
        <tr
          className={
            cptInfo["Removed rows"].includes(rowIdx)
              ? "highlight main-row-width"
              : "main-row-width"
          }
          key={rowIdx}
        >
          <td className={rowClassName}>
            {Utils.roundValue(row[rowLabels[0]])}
          </td>
          <td className={rowClassName}>
            {Utils.roundValue(row[rowLabels[1]], dataIsKPa)}
          </td>
          <td className={rowClassName}>
            {Utils.roundValue(row[rowLabels[2]], dataIsKPa)}
          </td>
          <td className={rowClassName}>
            {Utils.roundValue(row[rowLabels[3]], dataIsKPa)}
          </td>
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
              <th className="col-size-end" scope="col">
                u (MPa)
              </th>
            </tr>
          </thead>
          <tbody className="tbl-width scroll-tbl">
            <tr>
              <td className="tbl-width" colSpan="4">
                <div className=" scroll-tbl add-overlap-scrollbar">
                  <table className="tbl-width">
                    <tbody className="table-full-width">{cptTableRows}</tbody>
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
                  <td className="text-size">
                    {Utils.roundValue(cptInfo["z_min"])}
                  </td>
                </tr>
                <tr>
                  <td className="bold">Max Depth (m)</td>
                  <td className="text-size">
                    {Utils.roundValue(cptInfo["z_max"])}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="removed-info col-6 center-elm">
            <table className=" table thead-dark table-striped table-bordered mt-2 w-auto">
              <tbody>
                <tr>
                  <td className="bold info-width rem-label">
                    Depth Spread (m)
                  </td>
                  <td className="text-size">
                    {Utils.roundValue(cptInfo["z_spread"])}
                  </td>
                </tr>
                <tr className="highlight">
                  <td className="bold info-width">
                    <div className="row two-colum-row info-width">
                      <div className="col-9 rem-label">Removed Rows</div>
                      <div className=" col-1 file-info-tbl">
                        <InfoTooltip text={CONSTANTS.CPT_REMOVED_ROWS} />
                      </div>
                    </div>
                  </td>
                  <td className="text-size">
                    {cptInfo["Removed rows"].length}
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

export default memo(CPTTable);
