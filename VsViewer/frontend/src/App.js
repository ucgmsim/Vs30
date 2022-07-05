import React, {Fragment} from "react";
import { Tabs, Tab } from "react-bootstrap";

import { CPT, SPT, VsProfile } from "./components";

import './App.css';
// import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  return (
    <div className="App d-flex flex-column h-100">
      <Fragment>
        <Tabs defaultActiveKey="cpt" className="vs-tabs">
          <Tab
            eventKey="cpt"
            title="CPT"
            tabClassName="tab-fonts"
          >
            <CPT/>
          </Tab>

          <Tab
            eventKey="spt"
            title="SPT"
            tabClassName="tab-fonts"
          >
            <SPT/>
          </Tab>

          <Tab
            eventKey="vsprofile"
            title="Vs Profile"
            tabClassName="tab-fonts"
          >
            <VsProfile/>
          </Tab>
        </Tabs>
      </Fragment>
    </div>
  );
}

export default App;
