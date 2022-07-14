import React, { memo } from "react";

import "assets/cptTable.css";

const CPTTable = ({cptTableData}) => {
    if (cptTableData !== undefined) {
        const cptTableRows = [];
        cptTableData.forEach((row, rowIdx) => {
            const rowClassName = "col-size"
            cptTableRows.push(
                <tr key={rowIdx}>
                  <td className={rowClassName} >{row["Depth"]}</td>
                  <td className={rowClassName} >{row["Qc"]}</td>
                  <td className={rowClassName} >{row["Fs"]}</td>
                  <td className={rowClassName} >{row["u"]}</td>
                </tr>
              );
        });
    
        return (
            <table className="table thead-dark table-striped table-bordered mt-2 w-auto">
                <thead>
                    <tr>
                    <th className="col-size" scope="col">Depth</th>
                    <th className="col-size" scope="col">Qc</th>
                    <th className="col-size" scope="col">Fs</th>
                    <th className="col-size" scope="col">u</th>
                    </tr>
                </thead>
                <tbody className="tbl-width scroll-tbl">
                    <tr>
                        <td className="tbl-width" colSpan="4">
                            <div className="scroll-tbl">
                                <table className="tbl-width">{cptTableRows}</table>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>
        );
    }
};

export default memo(CPTTable);
