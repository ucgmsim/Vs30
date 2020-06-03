// map layer ids on server
var ID_GEOCAT = "aak_map_fill"


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

    var styles = document.getElementById('menu_mapstyle')
        .getElementsByTagName('a');
    for (var i = 0; i < styles.length; i++) {
        styles[i].onclick = switch_layer;
    }

    map.on("click", map_mouseselect);
}


function map_mouseselect(e) {
    var features = map.queryRenderedFeatures(e.point);
    var geocat;
    for (var i=0; i < features.length; i++) {
        if (features[i].layer.id === ID_GEOCAT && geocat === undefined) {
            geocat = features[i].properties.g;
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


$(document).ready(function ()
{
    load_map();
});
