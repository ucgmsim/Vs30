import React from "react";
import { Tabs, Tab } from "react-bootstrap";

import { CPT, SPT, VsProfile } from "components";
import { GlobalContextProvider } from "context";

import "assets/App.css";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  return (
    <GlobalContextProvider>
      <div className="App d-flex flex-column h-100">
        <Tabs defaultActiveKey="cpt" className="vs-tabs">
          <Tab eventKey="cpt" title="CPT" tabClassName="tab-fonts">
            <CPT />
          </Tab>

          <Tab eventKey="spt" title="SPT" tabClassName="tab-fonts">
            <SPT />
          </Tab>

          <Tab eventKey="vsprofile" title="Vs Profile" tabClassName="tab-fonts">
            <VsProfile />
          </Tab>
        </Tabs>
      </div>
    </GlobalContextProvider>
  );
}

export default App;
