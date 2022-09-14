import React, { createContext, useState } from "react";

export const Context = createContext({});

export const Provider = (props) => {
  const { children } = props;

  // CPT Data
  const [cptData, setCPTData] = useState({});
  const [cptMidpointData, setCptMidpointData] = useState({});
  const [cptWeights, setCptWeights] = useState({});
  const [cptResults, setCptResults] = useState({});
  const [cptCorrelationWeights, setCptCorrelationWeights] = useState({});

  // VsProfile
  const [vsProfileData, setVsProfileData] = useState([]);
  const [vsProfileMidpointData, setVsProfileMidpointData] = useState({});
  const [vsProfileWeights, setVsProfileWeights] = useState({});
  const [vsProfileResults, setVsProfileResults] = useState({});

  // SPT Data
  const [sptData, setSPTData] = useState({});
  const [sptMidpointData, setSptMidpointData] = useState({});
  const [sptWeights, setSptWeights] = useState({});
  const [sptResults, setSptResults] = useState({});
  const [sptCorrelationWeights, setSptCorrelationWeights] = useState({});

  // Make the context object:
  const globalContext = {
    // CPT Data
    cptData,
    setCPTData,
    cptWeights,
    setCptWeights,
    cptResults,
    setCptResults,
    cptCorrelationWeights,
    setCptCorrelationWeights,

    // VsProfile Data
    vsProfileData, 
    setVsProfileData,
    vsProfileWeights,
    setVsProfileWeights,
    vsProfileResults,
    setVsProfileResults,

    // Plotting
    cptMidpointData,
    setCptMidpointData,
    sptMidpointData,
    setSptMidpointData,
    vsProfileMidpointData,
    setVsProfileMidpointData,

    // SPT Data
    sptData,
    setSPTData,
    sptWeights,
    setSptWeights,
    sptResults,
    setSptResults,
    sptCorrelationWeights,
    setSptCorrelationWeights,
  };

  // pass the value in provider and return
  return <Context.Provider value={globalContext}>{children}</Context.Provider>;
};

export const { Consumer } = Context;
