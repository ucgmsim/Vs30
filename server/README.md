# Interactive web interface for Vs30

## Requirements
* *everything for Vs30*
* static webserver

## Mapbox Service Requirements
Map data is hosted by Mapbox which combines standard map tiles with our datasets. Preparation of this data is in the [data](data) directory.

## Running
The website is static. You can easily test/run with busybox:
```
busybox httpd -f -p 5088 -h ./webroot
```
Alternatively:
```
cd ./webroot && python -m http.server 5088
```
Or:
```
ruby -run -ehttpd ./webroot -p5088
```
And then access it at [http://localhost:5088](http://localhost:5088).
