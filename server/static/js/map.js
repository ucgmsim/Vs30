// map layer ids on server
var ID_GEOCAT = "aak_map_fill"
var ID_VSPR = "vspr"


function load_map()
{
    mapboxgl.accessToken = 'pk.eyJ1IjoiLXZpa3Rvci0iLCJhIjoiY2pzam9mNXVoMm9xdzQ0b2FmNnNqODE4NCJ9.AnNONHzKRb5vdl2Ikw2l2Q';
    map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/-viktor-/ckartgw8p3x071jlpifdt7h31',
        center: [173.2995, -41.2728],
        zoom: 5,
    });

    // zoom and rotation controls
    map.addControl(new mapboxgl.NavigationControl({visualizePitch: true}));
    // distance scale
    map.addControl(new mapboxgl.ScaleControl({maxWidth: 200, unit: 'metric'}), 'bottom-right');
    // popup not shown yet
    popup = new mapboxgl.Popup({closeButton: true});

    var styles = document.getElementById('menu_mapstyle')
        .getElementsByTagName('a');
    for (var i = 0; i < styles.length; i++) {
        styles[i].onclick = switch_layer;
    }

    map.on("click", map_mouseselect);
    map.on('mousemove', ID_VSPR, function(e) {
        map.getCanvas().style.cursor = 'pointer';
    });
    map.on('click', ID_VSPR, function(e) {
        var feature = e.features[0];
        popup.setLngLat(feature.geometry.coordinates)
            .setHTML('<strong>Site: ' + feature.properties.StationID + '</strong><p><table class="table table-sm"><tbody>' +
                //'<tr><th scope="row">Easting</th><td>' + feature.properties.Easting + '</td></tr>' +
                //'<tr><th scope="row">Northing</th><td>' + feature.properties.Northing + '</td></tr>' +
                '<tr><th scope="row">Vs30 (m/s)</th><td>' + feature.properties.Vs30 + '</td></tr>' +
                '<tr><th scope="row">lnMeasUncer</th><td>' + feature.properties.lnMeasUncer + '</td></tr>' +
                '<tr><th scope="row">Quality Flag</th><td>' + feature.properties.QualityFlag + '</td></tr>' +
                '<tr><th scope="row">AhdiAK Vs30</th><td>' + feature.properties.Vs30_AhdiAK_noQ3_hyb09c + '</td></tr>' +
                '<tr><th scope="row">AhdiAK stdev</th><td>' + feature.properties.stDv_AhdiAK_noQ3_hyb09c + '</td></tr>' +
                '<tr><th scope="row">YongCA Vs30</th><td>' + feature.properties.Vs30_YongCA_noQ3 + '</td></tr>' +
                '<tr><th scope="row">YongCA stdev</th><td>' + feature.properties.stDv_YongCA_noQ3 + '</td></tr>' +
                '</tbody></table></p>')
            .addTo(map);
        });
    map.on('mouseleave', ID_VSPR, function() {
        map.getCanvas().style.cursor = '';
    });
}


function map_mouseselect(e) {
    var features = map.queryRenderedFeatures(e.point);
    var geocat;
    for (var i=0; i < features.length; i++) {
        if (features[i].layer.id === ID_GEOCAT && geocat === undefined) {
            geocat = features[i].properties.gid;
        }
    }
    console.log(geocat)
}


function switch_layer(layer) {
    document.getElementById("menu_mapstyle").getElementsByClassName("active")[0].classList.remove("active");
    map.setStyle('mapbox://styles/' + layer.target.id);
    layer.target.classList.add("active");
}


var map;
var popup;


$(document).ready(function ()
{
    load_map();
});
