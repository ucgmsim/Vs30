
function load_map()
{
    mapboxgl.accessToken = 'pk.eyJ1IjoiLXZpa3Rvci0iLCJhIjoiY2pzam9mNXVoMm9xdzQ0b2FmNnNqODE4NCJ9.AnNONHzKRb5vdl2Ikw2l2Q';
    map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/streets-v11',
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

    map.on("load", map_loaded);
}

function map_loaded()
{
    var layers = map.getStyle().layers;
    // Find the index of the first symbol layer in the map style
    var first_symbol_id;
    for (var i=0; i<layers.length; i++) {
        if (layers[i].type === 'symbol') {
            first_symbol_id = layers[i].id;
            break;
        }
    }
    map.addSource(
        "aak_map", {
        "type": "geojson",
        "data": _static + "geo/aak_map.geojson"
    });
    map.addLayer({
        "id": "aak_map",
        "type": "fill",
        "source": "aak_map",
        "layout": {},
        "paint": {
            "fill-color": ["match", ["get", "g"],
            "1", "#FFB080",
            "2", "#FFE080",
            "3", "#EEFF80",
            "4", "#BFFF80",
            "5", "#8FFF80",
            "6", "#80FFA1",
            "7", "#80FFD1",
            "8", "#80FFFF",
            "9", "#80CFFF",
            "10", "#80A1FF",
            "11", "#9180FF",
            "12", "#BF80FF",
            "13", "#F080FF",
            "14", "#FF80DE",
            "15", "#FF80AE",
            "#000" // other
            ],
            "fill-opacity": 0.8
        }
    }, first_symbol_id);
}

function map_styled(e) {
    console.log(map.loaded());
    if (map.loaded() && map.getSource("aak_map") === undefined) map_loaded();
}

function switch_layer(layer) {
    document.getElementById("menu_mapstyle").getElementsByClassName("active")[0].classList.remove("active");
    map.setStyle('mapbox://styles/mapbox/' + layer.target.id);
    layer.target.classList.add("active");
}

var map;

$(document).ready(function ()
{
    load_map();
});
