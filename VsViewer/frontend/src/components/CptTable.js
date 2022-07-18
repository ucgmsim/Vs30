import React, { memo } from "react";

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
          <td className={rowClassName}>{row["Depth"]}</td>
          <td className={rowClassName}>{row["Qc"]}</td>
          <td className={rowClassName}>{row["Fs"]}</td>
          <td className={rowClassName}>{row["u"]}</td>
        </tr>
      );
    });

    return (
      <div>
        <table className="cpt-raw table thead-dark table-striped table-bordered mt-2 w-auto">
          <thead>
            <tr>
              <th className="col-size" scope="col">
                Depth
              </th>
              <th className="col-size" scope="col">
                Qc
              </th>
              <th className="col-size" scope="col">
                Fs
              </th>
              <th className="col-size" scope="col">
                u
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
                  <td className="bold">Min Depth</td>
                  <td>{cptInfo["z_min"]}m</td>
                </tr>
                <tr>
                  <td className="bold">Max Depth</td>
                  <td>{cptInfo["z_max"]}m</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="removed-info col-6 center-elm">
            <table className=" table thead-dark table-striped table-bordered mt-2 w-auto">
              <tbody>
                <tr>
                  <td className="bold">Depth Spread</td>
                  <td>{cptInfo["z_spread"]}m</td>
                </tr>
                <tr className="highlight">
                  <td className="bold">Removed Rows</td>
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
