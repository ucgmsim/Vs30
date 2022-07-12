import React, { createContext, useState } from "react";

export const Context = createContext({});

export const Provider = (props) => {
    const { children } = props;
  
    // CPT Data
    const [cptData, setCPTData] = useState([]);
    const cptMidpointData = {};
    const cptPlotData = {};

    // Make the context object:
    const globalContext = {
         // CPT Data
         cptData,
         setCPTData,

         // Plotting
         cptMidpointData,
         cptPlotData,
    };

    // pass the value in provider and return
    return <Context.Provider value={globalContext}>{children}</Context.Provider>;
};

export const { Consumer } = Context;