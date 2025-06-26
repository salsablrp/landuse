let mainMap = null;
let currentElement = "";
 
$("input[type=textbox]").focus(function() {
    currentElement = $(this).attr("id");
}); 

$("#distance-slider").on("input", function(){
    $('#val').text($("#distance-slider")[0].value);
    drawCircle();
});
 
function drawCircle(){
    centroid = $("#" + currentElement).val().split(",");
    centroid = [parseFloat(centroid[0]), parseFloat(centroid[1])];
    radious = parseFloat($("#val").text());
 
    let circle = new ol.geom.Circle(centroid, radious * 1000.0);
 
    const vectorSource = new ol.source.Vector({
        features: [new ol.Feature(circle)],
    });
      
    const vectorLayer = new ol.layer.Vector({
        name: "circle",
        source: vectorSource,
        style: new ol.style.Style({
            stroke: new ol.style.Stroke({
                color: '#ff0000',
                width: 2,
            }),
            fill: new ol.style.Fill({
                color: 'rgba(255, 255, 255, 0.4)'
            })
        })
    });
 
    removeLayerByName(mainMap, "circle");
    mainMap.addLayer(vectorLayer);
}

function init() {
    // Define the map view
    let mainView = new ol.View({
        center: [0, 0],  // temporary center
        zoom: 2          // global zoom level
    });

    // Initialize the map
    mainMap = new ol.Map({
        controls: [],
        target: 'map', /* Set the target to the ID of the map */
        view: mainView,
        controls: []
    });

    const popupElement = document.createElement('div');
    popupElement.id = 'popup';
    document.body.appendChild(popupElement);

    const popupOverlay = new ol.Overlay({
        element: popupElement,
        positioning: 'bottom-center',
        stopEvent: false
    });

    mainMap.addOverlay(popupOverlay);

    let baseLayer = getBaseMap("osm");

    mainMap.addLayer(baseLayer);

    // Handle the change in the measuretype dropdown
    document.getElementById('measuretype').addEventListener('change', function (event) {
        let measureOption = event.target.value;

        switch (measureOption) {
            case 'length':
                // Start drawing for Length measurement (LineString)
                startDrawing('LineString');
                break;
            case 'area':
                // Start drawing for Area measurement (Polygon)
                startDrawing('Polygon');
                break;
            case 'clear':
                // Clear current measurement
                clearMeasurement();
                break;
            default:
                break;
        }
    });

    mainMap.on('click', function(evt) {
        let val = evt.coordinate[0].toString() + "," + evt.coordinate[1].toString();
        if (currentElement != "") {
            $("#" + currentElement).val(val);

            let name = "location_1";
            let color = "#FF0000";

            if (currentElement == 'end') {
                name = "location_2";
                color = "#00FF00";
            }

            const feature = new ol.Feature({
                geometry: new ol.geom.Point([evt.coordinate[0], evt.coordinate[1]]),
            });

            feature.setStyle(
                new ol.style.Style({
                    image: new ol.style.Icon({
                        color: color,
                        src: './images/pin.svg',
                        width: 30,
                    })
                })
            );

            const layer = new ol.layer.Vector({
                name: name,
                source: new ol.source.Vector({
                    features: [feature],
                })
            });
            layer.setZIndex(100);

            removeLayerByName(mainMap, name);
            mainMap.addLayer(layer);

            if (currentElement == "location-search") {
                drawCircle();
            }
        }
    });
}

// Function to create a base map layer
function getBaseMap(name) {
    let baseMaps = {
        "osm": {
            url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            attributions: ''
        },
        "otm": {
            url: 'https://b.tile.opentopomap.org/{z}/{x}/{y}.png',
            attributions: 'Kartendaten: © OpenStreetMap-Mitwirkende, SRTM | Kartendarstellung: © OpenTopoMap (CC-BY-SA)'
        },
        "esri_wtm": {
            url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            attributions: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community'
        },
        "esri_natgeo": {
            url: 'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
            attributions: 'Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC'
        }
    }

    let layer = baseMaps[name];
    if (layer === undefined) {
        layer = baseMaps["osm"]
    }

    return (
        new ol.layer.Tile({
            name: "base",
            source: new ol.source.TileImage(layer)
        })
    )
}

function hidePanels(){
    $(".panel").hide();
    $(".alert").hide();
    clearResults();
}

function clearResults(){
    $("input[type=textbox]").val("");
    currentElement = "";
    layers = ["location_1", "location_2", "route", "markets", "area", "circle"];
    for (let i = 0; i < layers.length; i++)
        removeLayerByName(mainMap, layers[i]);
}
 
function showPanel(id){
    hidePanels();
    $("#" + id).show();
}
 
$('.close-icon').on('click',function() {
    $(this).closest('.card').fadeOut();
})

function removeLayerByName(map, layer_name){
    let layerToRemove = null;
    map.getLayers().forEach(function (layer) {
        if (layer.get('name') != undefined && layer.get('name') === layer_name) {
            layerToRemove = layer;
        }
    });
 
    map.removeLayer(layerToRemove);
}
 
$("input[name=basemap]").click(function(evt){
    removeLayerByName(mainMap, "base");
    let baseLayer = getBaseMap(evt.target.value);
    mainMap.addLayer(baseLayer);    
});

$("#btnService").click(function(){
    removeLayerByName(mainMap, "area");
    $("#pnl-service-alert").hide();
 
    $.ajax({
        url: "./services/service_area.py?" +
            "location=" + $("#location-service").val() +
            "&size=" + $("input[name=size]:checked")[0].value +
            "&srid=3857",
        type: "GET",
        success: function(data){
            if (data.length != 0){
                let vectorLayer = new ol.layer.Vector({
                    name: "area",
                    source: new ol.source.Vector({
                        features: new ol.format.GeoJSON().readFeatures(data[0].geom),
                    }),
                    style: new ol.style.Style({
                        stroke: new ol.style.Stroke({
                            color: '#ff0000',
                            width: 2,
                        }),
                        fill: new ol.style.Fill({
                            color: 'rgba(255, 255, 255, 0.4)'
                        })
                    })
                });
 
                mainMap.addLayer(vectorLayer);
            } 
        },
        error: function(data){
            $("#pnl-service-alert").html("Error: An error occurred while executing the tool.");
            $("#pnl-service-alert").show();
        }
    })
});

const sizes = {
    "small_markets": 15,
    "local_markets": 20,
    "medium_markets": 25,
    "capital_markets": 40
};
 
$("#btnSearch").click(function(){
    removeLayerByName(mainMap, "districts");
    $("#pnl-search-alert").hide();

    const level = document.getElementById("district").value;
    if (!level) return;

    const url = `data/admin_level${level}.geojson`;

    const source = new ol.source.Vector({
        url: url,
        format: new ol.format.GeoJSON()
    });

    const layer = new ol.layer.Vector({
        name: "districts",
        source: source,
        style: new ol.style.Style({
            stroke: new ol.style.Stroke({ color: '#333', width: 1.5 }),
            fill: new ol.style.Fill({ color: 'rgba(100, 100, 255, 0.2)' })
        })
    });

    mainMap.addLayer(layer);

    source.once('change', () => {
        if (source.getState() === 'ready') {
            mainMap.getView().fit(source.getExtent(), { padding: [20, 20, 20, 20] });
        }
    });
});

$("#btnRoute").click(function () {
    // Remove any previous change layers
    removeLayerByName(mainMap, "lu_change_tif");
    $("#pnl-route-alert").hide();

    // Load the raster layer
    const rasterLayer = new ol.layer.Tile({
        name: "lu_change_tif",
        source: new ol.source.TileImage({
            url: 'data/lu_change.tif',
            tileLoadFunction: function (imageTile, src) {
                imageTile.getImage().src = src;
            }
        })
    });

    mainMap.addLayer(rasterLayer);

    source.once('change', () => {
        if (source.getState() === 'ready') {
            mainMap.getView().fit(source.getExtent(), { padding: [20, 20, 20, 20] });
        }
    });
});

$("#btnClosest").click(function () {
    removeLayerByName(mainMap, "predicted");
    $("#pnl-closest-alert").hide();

    const layerName = document.getElementById("layer").value;
    const attribute = document.getElementById("attributes").value;
    const operator = document.getElementById("operator").value;
    const value = document.getElementById("value").value;

    const layerObj = analysisLayers[layerName];
    if (!layerObj) return;

    const features = layerObj.getSource().getFeatures().filter(f => {
        const val = f.get(attribute);
        if (operator === '=') return val == value;
        if (operator === '>') return val > value;
        if (operator === '<') return val < value;
        return false;
    });

    const filteredLayer = new ol.layer.Vector({
        name: "predicted",
        source: new ol.source.Vector({ features: features }),
        style: new ol.style.Style({
            fill: new ol.style.Fill({ color: 'rgba(255, 255, 0, 0.6)' }),
            stroke: new ol.style.Stroke({ color: '#ff0', width: 2 })
        })
    });

    mainMap.addLayer(filteredLayer);

    filteredLayer.getSource().once('change', () => {
        if (filteredLayer.getSource().getState() === 'ready') {
            mainMap.getView().fit(filteredLayer.getSource().getExtent(), { padding: [20, 20, 20, 20] });
        }
    });
});

////////////////////////////// YEAR SLIDER //////////////////////////////
const startSlider = document.getElementById('start-year');
const endSlider = document.getElementById('end-year');
const startVal = document.getElementById('start-val');
const endVal = document.getElementById('end-val');
// Update the start year value display when the start slider changes
startSlider.addEventListener('input', function() {
    startVal.textContent = startSlider.value;
});
// Update the end year value display when the end slider changes
endSlider.addEventListener('input', function() {
    endVal.textContent = endSlider.value;
});
// Ensure that the start year is always less than or equal to the end year
startSlider.addEventListener('input', function() {
    if (parseInt(startSlider.value) > parseInt(endSlider.value)) {
        endSlider.value = startSlider.value;
        endVal.textContent = endSlider.value;
    }
});
endSlider.addEventListener('input', function() {
    if (parseInt(endSlider.value) < parseInt(startSlider.value)) {
        startSlider.value = endSlider.value;
        startVal.textContent = startSlider.value;
    }
});

////////////////////////////// LEGEND //////////////////////////////
var overlays = new ol.layer.Group({
    'title': 'Overlays',
    layers: []
});

function show_hide_legend() {

    if (document.getElementById("legend").style.visibility == "hidden") {

        document.getElementById("legend_btn").innerHTML = "☰ Hide Legend";
		document.getElementById("legend_btn").setAttribute("class", "btn btn-danger btn-sm");

        document.getElementById("legend").style.visibility = "visible";
        document.getElementById("legend").style.width = "15%";

        document.getElementById('legend').style.height = '38%';
        map.updateSize();
    } else {
	    document.getElementById("legend_btn").setAttribute("class", "btn btn-success btn-sm");
        document.getElementById("legend_btn").innerHTML = "☰ Show Legend";
        document.getElementById("legend").style.width = "0%";
        document.getElementById("legend").style.visibility = "hidden";
        document.getElementById('legend').style.height = '0%';

        map.updateSize();
    }
}

function legend() {
    $('#legend').empty();
    var no_layers = overlays.getLayers().get('length');
    //console.log(no_layers[0].options.layers);
    // console.log(overlays.getLayers().get('length'));
    //var no_layers = overlays.getLayers().get('length');

    var head = document.createElement("h8");

    var txt = document.createTextNode("Legend");

    head.appendChild(txt);
    var element = document.getElementById("legend");
    element.appendChild(head);


    overlays.getLayers().getArray().slice().forEach(layer => {

        var head = document.createElement("p");

        var txt = document.createTextNode(layer.get('title'));
        //alert(txt[i]);
        head.appendChild(txt);
        var element = document.getElementById("legend");
        element.appendChild(head);
        var img = new Image();
        img.src = "http://localhost:8084/geoserver/wms?REQUEST=GetLegendGraphic&VERSION=1.0.0&FORMAT=image/png&WIDTH=20&HEIGHT=20&LAYER=" + layer.get('title');
        var src = document.getElementById("legend");
        src.appendChild(img);

    });
}
legend();

////////////////////////////// MEASURE TOOL //////////////////////////////
// Variables for drawing interactions
let draw; // global so we can remove it later

// Function to start drawing (line or polygon)
function startDrawing(type) {
    // Remove previous drawing interaction if any
    if (draw) {
        mainMap.removeInteraction(draw);
    }

    // Define the drawing interaction
    draw = new ol.interaction.Draw({
        source: new ol.source.Vector(),
        type: type,
    });

    // Add event listener for drawing end to calculate length or area
    draw.on('drawend', function (event) {
        if (type === 'LineString') {
            let length = formatLength(event.feature.getGeometry());
            alert('Length: ' + length);
        } else if (type === 'Polygon') {
            let area = formatArea(event.feature.getGeometry());
            alert('Area: ' + area);
        }
    });

    // Add the draw interaction to the map
    mainMap.addInteraction(draw);
}

// Function to format the length of a line
function formatLength(geometry) {
    let length = ol.sphere.getLength(geometry);
    return (length).toFixed(2) + ' m'; // Convert to meters and round
}

// Function to format the area of a polygon
function formatArea(geometry) {
    let area = ol.sphere.getArea(geometry);
    return (area).toFixed(2) + ' m²'; // Convert to square meters and round
}

// Function to clear measurement
function clearMeasurement() {
    // Remove the drawing interaction if any
    if (draw) {
        mainMap.removeInteraction(draw);
    }

    // Optionally, clear the vector layer where the shapes are stored
    mainMap.getLayers().forEach(function(layer) {
        if (layer instanceof ol.layer.Vector) {
            layer.getSource().clear();
        }
    });

    // Reset the measuretype dropdown to default
    document.getElementById('measuretype').value = 'select';
}

////////////////// ADMIN LEVEL //////////////////

// Load Files
const adminLevels = {
    0: 'data/admin_level0.geojson',
    1: 'data/admin_level1.geojson',
    2: 'data/admin_level2.geojson'
  };
  
let adminLayer = null;

function loadAdmin(level) {
if (adminLayer) map.removeLayer(adminLayer);
const src = new ol.source.Vector({
    url: adminLevels[level],
    format: new ol.format.GeoJSON()
});
adminLayer = new ol.layer.Vector({
    source: src,
    style: new ol.style.Style({
    stroke: new ol.style.Stroke({ color: '#555', width: 2 }),
    fill: new ol.style.Fill({ color: 'rgba(200,200,200,0.4)' })
    })
});
mainMap.addLayer(adminLayer);

src.on('addfeature', () => {
    mainMap.getView().fit(src.getExtent(), { padding: [20,20,20,20] });
});

mainMap.on('singleclick', function(evt) {
    mainMap.forEachFeatureAtPixel(evt.pixel, function(feat) {
    const props = feat.getProperties();
    delete props.geometry;
    const rows = Object.entries(props).map(e=>`<tr><th>${e[0]}</th><td>${e[1]}</td></tr>`).join('');
    popupOverlay.getElement().innerHTML = `<table class="table table-sm">${rows}</table>`;
    popupOverlay.setPosition(evt.coordinate);
    });
});
}

// Populate the dropdown and bind in init()
const sel = document.getElementById('district');
Object.keys(adminLevels).forEach(l => {
  const o = new Option('Level '+l, l);
  sel.add(o);
});
document.getElementById('btnSearch').onclick = () => {
    const level = sel.value;
    if (level !== '') {
        loadAdmin(level);
    }
};

////////////////// MONITORING //////////////////

// Load file
document.getElementById("btnRoute").addEventListener("click", function () {
    removeLayerByName(mainMap, "lu_change_tif");
    document.getElementById("pnl-route-alert").style.display = "none";

    const changeLayer = new TileLayer({
      name: "lu_change_tif",
      source: new GeoTIFF({
        sources: [{ url: 'data/lu_change.tif' }],
        normalize: false
      })
    });

    mainMap.addLayer(changeLayer);
  
    src2.once('change', () => {
      if (src2.getState() === 'ready') {
        mainMap.getView().fit(src2.getExtent(), { padding: [20, 20, 20, 20] });
      }
    });
  };  

////////////////// PREDICTIVE MODEL //////////////////

// Define Sample layers
const ops = ['=', '>', '<'];

// Populate dropdowns dynamically
const lay = document.getElementById('layer');
const attr = document.getElementById('attributes');
const op = document.getElementById('operator');

Object.keys(analysisLayers).forEach(n => lay.add(new Option(n, n)));
lay.onchange = () => {
  const lyr = analysisLayers[lay.value];
  const f = lyr.getSource().getFeatures()[0];
  Object.keys(f.getProperties()).filter(p=>p!=='geometry').forEach(k => attr.add(new Option(k,k)));
};
ops.forEach(o => operator.add(new Option(o, o)));

// Apply filter and highlights
document.getElementById('btnClosest').onclick = () => {
    const analysisLayers = {
        'Land Use 2024': map2024,
        'Districts': adminLayer
      };
    const lyr = analysisLayers[lay.value];
    const a = attr.value, o = operator.value, v = document.getElementById('value').value;
    const feat = lyr.getSource().getFeatures().filter(f => {
      const val = f.get(a);
      return o==='=' ? val==v : o==='>' ? val>v : val<v;
    });
    // Highlight them:
    const selStyle = new ol.style.Style({ fill: new ol.style.Fill({ color:'rgba(255,255,0,0.6)' }), stroke: new ol.style.Stroke({ color:'#ff0', width:2 }) });
    feat.forEach(f=>f.setStyle(selStyle));
  };