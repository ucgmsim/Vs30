# Interactive web interface for Vs30

## Requirements
* *everything for Vs30*
* uwsgi (with python, http support)
* flask

## Mapbox Service Requirements
Map data is hosted by Mapbox which combines standard map tiles with our datasets. Preparation of this data is in the [data](data) directory.

## Running
Simply execute `run.sh`.
This script may need modification of `uwsgi` parameters based on your system configuration.
The server runs on port 5088 [http://localhost:5088](http://localhost:5088).
