import React, { memo, useState, useEffect } from "react";

import "assets/weights.css";

const WeightTable = ({weights, setFunction, refresh}) => {

    const changeWeightRows = (displayWeights) => {
        let weightRows = [];
        for (const key of Object.keys(displayWeights)) {
            weightRows.push(
                <tr key={key}>
                    <td className="bold col-size" >{key}</td>
                    <td>
                        <input className="col-size" value={weights[key]} onChange={(e) => changeWeight(e.target.value, key)}></input>
                    </td>
                </tr>
              );
        };
        return weightRows;
    };

    const [weightRows, setWeightRows] = useState(changeWeightRows(weights));

    // Change the Parent Weights
    const changeWeight = (entry, key) => {
        weights[key] = entry
        setFunction(weights);
        setWeightRows(changeWeightRows(weights));
    }

    useEffect(() => {
        if (weights !== undefined) {
            setWeightRows(changeWeightRows(weights));
        }
    }, [weights]);

    if (weights !== undefined) {
        return (
            <div className="scroll-weights table-padding">
                <table className="table thead-dark table-striped table-bordered weight-table">
                    <tbody>
                        {weightRows}
                    </tbody>
                </table>
            </div>
        );
    }
};

export default memo(WeightTable);
