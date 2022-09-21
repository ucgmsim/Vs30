import React from "react";

import "assets/fileTable.css";

const FileTable = ({ files, removeFunction }) => {
  const changeLength = (key) => {
    // Ensures length of the keys are not greater than 15 characters
    if (key.length > 15) {
      return key.substring(0, 12) + "...";
    } else {
      return key;
    }
  };

  if (files !== undefined && files !== null) {
    const tableRows = [];
    files.forEach((row, rowIdx) => {
      tableRows.push(
        <tr className="row-file-table" key={rowIdx}>
          <td className="file-col-size">{changeLength(row["label"])}</td>
          <td className="remove-col-size">
            <button
              type="button"
              className="btn-close btn-size"
              onClick={(row) => removeFunction(files[rowIdx])}
            />
          </td>
        </tr>
      );
    });

    return (
      <div className="scroll-file-tbl centre-elm remove-scrollbar">
        <table className="table file-table table-bordered">
          <tbody>{tableRows}</tbody>
        </table>
      </div>
    );
  }
};

export default FileTable;
