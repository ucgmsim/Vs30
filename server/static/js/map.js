// map layer ids on server
var ID_GEOCAT = "aak_map_fill"
var ID_GEOV30 = "gvs30_map_fill"
var ID_GEOSDV = "gstdv_map_fill"
var ID_TERCAT = "ip_map_fill"
var ID_TERV30 = "tvs30_map_fill"
var ID_TERSDV = "tstdv_map_fill"
var ID_COMV30 = "cvs30_map_fill"
var ID_COMSDV = "cstdv_map_fill"
var ID_VSPR = "vspr"

var NAME_GEOCAT = [
    "0: Water",
    "1: Peat",
    "4: Fill",
    "5: Fluvial Estuarine",
    "6: Alluvium",
    "8: Lacustrine",
    "9: Beach Bar Dune",
    "10: Fan",
    "11: Loess",
    "12: Outwash",
    "13: Floodplain",
    "14: Moraine Till",
    "15: Undif Sed",
    "16: Terrace",
    "17: Volcanic",
    "18: Crystalline"
];
var NAME_TERCAT = [
    "1: Well dissected alpine summits, mountains, etc.",
    "2: Large volcano, high block plateaus, etc.",
    "3: Well dissected, low mountains, etc.",
    "4: Volcanic fan, foot slope of high block plateaus, etc.",
    "5: Dissected plateaus, etc.",
    "6: Basalt lava plain, glaciated plateau, etc.",
    "7: Moderately eroded mountains, lava flow, etc.",
    "8: Desert alluvial slope, volcanic fan, etc.",
    "9: Well eroded plain of weak rocks, etc.",
    "10: Valley, till plain, etc.",
    "11: Eroded plain of weak rocks, etc.",
    "12: Desert plain, delta plain, etc.",
    "13: Incised terrace, etc.",
    "14: Eroded alluvial fan, till plain, etc.",
    "15: Dune, incised terrace, etc.",
    "16: Fluvial plain, alluvial fan, low-lying flat plains, etc."
]


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
    // marker will be used whin location doesn't update with mousemove
    marker = new mapboxgl.Marker()

    // control for layer to be visible
    var layers = document.getElementById('menu_layer')
        .getElementsByTagName('a');
    for (var i = 0; i < layers.length; i++) {
        layers[i].onclick = switch_layer;
    }

    map.on("click", map_mouseselect);
    map.on("mousemove", map_mouseselect);
    map.on("mousemove", ID_VSPR, function(e) {
        map.getCanvas().style.cursor = 'pointer';
    });
    map.on('click', ID_VSPR, function(e) {
        var feature = e.features[0];
        new mapboxgl.Popup({closeButton: true}).setLngLat(feature.geometry.coordinates)
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


function follow_mouse(cb) {
    if (cb.checked) {
        map.on("mousemove", map_mouseselect);
    } else {
        map.off("mousemove", map_mouseselect);
    }
}

function map_mouseselect(e) {
    var follow = document.getElementById("follow_mouse").checked;

    var features = map.queryRenderedFeatures(e.point);
    var geocat;
    var tercat;
    var vspr = false;
    for (var i=0; i < features.length; i++) {
        if (features[i].layer.id === ID_GEOCAT && geocat === undefined) {
            geocat = features[i].properties.gid;
        } else if (features[i].layer.id === ID_TERCAT && tercat === undefined) {
            tercat = features[i].properties.gid;
        } else if (features[i].layer.id === ID_VSPR) {
            if (! follow) {
                // opening station info shouldn't update location if using marker
                return;
            }
        }
    }

    // update UI
    document.getElementById("lon").value = e.lngLat.lng;
    document.getElementById("lat").value = e.lngLat.lat;
    if (! follow) {
        marker.setLngLat([e.lngLat.lng, e.lngLat.lat]).addTo(map);
        if (map.getZoom() < 10) {
            // can't see 100m grid
            map.flyTo({center: e.lngLat, zoom: 10});
        }
        if (! map.areTilesLoaded()) {
            document.getElementById("gid_aak").value = "loading...";
            document.getElementById("gid_yca").value = "loading...";
            // add event listener
            return;
        }
    }
    if (geocat === undefined) {
        document.getElementById("gid_aak").value = "NA";
    } else {
        document.getElementById("gid_aak").value = NAME_GEOCAT[geocat];
    }
    if (tercat === undefined) {
        document.getElementById("gid_yca").value = "NA";
    } else {
        document.getElementById("gid_yca").value = NAME_TERCAT[tercat - 1];
    }
}


function switch_layer(layer) {
    var old_element = document.getElementById("menu_layer").getElementsByClassName("active")[0]
    old_element.classList.remove("active");
    if (old_element.id !== "none") map.setPaintProperty(old_element.id, 'fill-opacity', 0);
    if (layer.target.id != "none") map.setPaintProperty(layer.target.id, 'fill-opacity', 0.8);
    layer.target.classList.add("active");
}


var map;
var marker;

$(document).ready(function ()
{
    load_map();
});
