# VsViewer - Frontend

## Contents
- [Overview](#overview)
- [Naming](#naming)
- [Requirements](#requirements)
- [Running locally](#running-locally)

## Overview

Frontend Web app to create Vs30 values from CPT, SPT and VsProfile data for a given site.

This is a React/Javascript SPA,  talking to a Python Flask API, running on a Linux host.


## Naming

- **Filename** : PascalCase (ex: CPT.js, VsProfile.js) except index files (ex: index.js, index.html, index.css...)
- **Variables & Functions** : camelCase (ex: requestOptions, formData) except the function is used to render a component (ex: CPT.js)
- **HTML Class/ID Names** : All lower case and separate with a hyphen (ex:hi-my-name-is-joel)

## Requirements

- Node v12
- `.env.dev` with the following environment variable.
  - REACT_APP_VS_API_URL=

#### To run Frontend: `npm start`

## Running locally

Open a terminal to do the following steps

1. Change the directory to frontend

```shell
cd /your_path/Vs30/VsViewer/frontend
```

2. Install packages

```shell
npm install
```

3. Start an app

```shell
npm start
```