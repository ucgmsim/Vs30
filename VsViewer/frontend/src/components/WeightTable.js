import React, { memo, useState, useEffect } from "react";

import "assets/weights.css";

const WeightTable = ({ weights, setFunction, flashError }) => {
  const changeLength = (key) => {
    // Ensures length of the keys are not greater than 15 char
    if (key.length > 15) {
      return key.substring(0, 12) + "...";
    } else {
      return key;
    }
  };

  const changeWeightRows = (displayWeights) => {
    let weightRows = [];
    for (const key of Object.keys(displayWeights)) {
      weightRows.push(
        <tr key={key}>
          <td className="bold weight-col-size">{changeLength(key)}</td>
          <td className="weight-col-size">
            <input
              className="weight-value"
              value={weights[key]}
              onChange={(e) => changeWeight(e.target.value, key)}
            ></input>
          </td>
        </tr>
      );
    }
    return weightRows;
  };

  const [weightRows, setWeightRows] = useState(changeWeightRows(weights));

  // Change the Parent Weights
  const changeWeight = (entry, key) => {
    weights[key] = entry;
    setFunction(weights);
    setWeightRows(changeWeightRows(weights));
  };

  useEffect(() => {
    if (weights !== undefined) {
      setWeightRows(changeWeightRows(weights));
    }
  }, [weights]);

  if (weights !== undefined) {
    return (
      <div className="scroll-weights table-padding remove-scrollbar">
        <table
          className={
            flashError
              ? "table thead-dark table-striped table-bordered weight-table flash-warning"
              : "table thead-dark table-striped table-bordered weight-table custom-border-size"
          }
        >
          <tbody>{weightRows}</tbody>
        </table>
      </div>
    );
  }
};

export default memo(WeightTable);
