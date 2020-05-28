
function load_map()
{
    mapboxgl.accessToken = 'pk.eyJ1IjoiLXZpa3Rvci0iLCJhIjoiY2pzam9mNXVoMm9xdzQ0b2FmNnNqODE4NCJ9.AnNONHzKRb5vdl2Ikw2l2Q';
    map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/light-v10',
        center: [173.2995, -41.2728],
        zoom: 5,
    });

    var styles = document.getElementById('menu_mapstyle')
        .getElementsByTagName('a');
    for (var i = 0; i < styles.length; i++) {
        styles[i].onclick = switch_layer;
    }
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
