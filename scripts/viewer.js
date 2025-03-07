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
        extent: [3124925, -599644, 3537136, -158022],
        center: [3336467, -385622],
        minZoom: 6,
        maxZoom: 14,
        zoom: 9
    });

    // Initialize the map
    mainMap = new ol.Map({
        controls: [],
        target: 'map', /* Set the target to the ID of the map */
        view: mainView,
        controls: []
    });

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
    removeLayerByName(mainMap, "markets");
    $("#pnl-search-alert").hide();
 
    $.ajax({
        url:"./services/search.py?location=" +
            $("#location-search").val() +
            "&distance=" +
            $("#val").text() +
            "&srid=3857",
        type: "GET",
        success: function(data){
            if (data.length != 0){
                let features = [];
                for (var i = 0; i < data.length; i++){    
                    var feature = new ol.format.GeoJSON().readFeature(data[i].geom);
                    feature.setStyle(
                        new ol.style.Style({ 
                            image: new ol.style.Icon({
                                src: './images/market.png',            
                                width: sizes[data[i].categorie]
                            })
                        })
                    );
                    features.push(feature);
                }
 
                const vectorSource = new ol.source.Vector({
                    features: features,
                })
 
                const vectorLayer = new ol.layer.Vector({
                    name: "markets",
                    source: vectorSource,
                })
                
                mainMap.addLayer(vectorLayer)
            } 
        },
        error: function(data){
            $("#pnl-search-alert").html("Error: An error occurred while executing the tool.");
            $("#pnl-search-alert").show();
        }
    })
});

$("#btnRoute").click(function () { 
    removeLayerByName(mainMap, "route");
    $("#pnl-route-alert").hide();
    
    $.ajax({
        url: "./services/routing.py?source=" + 
            $("#start").val() + 
            "&target=" + 
            $("#end").val() + 
            "&srid=3857", 
        type: "GET",
        success: function(data){
            if (data.path != null){
                let vectorLayer = new ol.layer.Vector({
                    name: "route",
                    source: new ol.source.Vector({
                        features: new ol.format.GeoJSON().readFeatures(data.path),
                    }),
                    style: new ol.style.Style({
                        stroke: new ol.style.Stroke({
                            color: '#ff0000',
                            width: 4,
                        }),
                    })
                });
                mainMap.addLayer(vectorLayer);
                
            } 
        },
        error: function(data){
            $("#pnl-route-alert").html("Error: An error occurred while executing the tool.");
            $("#pnl-route-alert").show();
        }
    })
});

$("#btnClosest").click(function () { 
	removeLayerByName(mainMap, "markets");
	$("#pnl-closest-alert").hide();
	
	$.ajax({
		url: "./services/closest_markets.py?location=" + 
			$("#location-closest").val() + 
			"&srid=3857",
		type: "GET",
		success: function(data){
			if (data.length != 0){
				let features = [];
				for (var i = 0; i < data.length; i++){
					var feature = new ol.format.GeoJSON().readFeature(data[i].geometry);
					feature.setStyle(
						new ol.style.Style({ 
							image: new ol.style.Icon({
								src: './images/market.png',			
								width: sizes[data[i].categorie]
							})
						})
					);
					features.push(feature);
				}

				const vectorSource = new ol.source.Vector({
					features: features,
				})

				const vectorLayer = new ol.layer.Vector({
					name: "markets",
					source: vectorSource,
				})
				
				mainMap.addLayer(vectorLayer)
			} 
		},
		error: function(data){
			$("#pnl-closest-alert").html("Error: An error occurred while executing the tool.");
			$("#pnl-closest-alert").show();
		}
	})
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
measuretype.onchange = function() {
    mainMap.un('singleclick', getinfo);
    document.getElementById("info_btn").innerHTML = "☰ Activate GetInfo";
    document.getElementById("info_btn").setAttribute("class", "btn btn-success btn-sm");
    if (popup) {
        popup.hide();
    }
    mainMap.removeInteraction(draw);
    addInteraction();
};


var source = new ol.source.Vector();
var vectorLayer = new ol.layer.Vector({
    //title: 'layer',
    source: source,
    style: new ol.style.Style({
        fill: new ol.style.Fill({
            color: 'rgba(255, 255, 255, 0.2)'
        }),
        stroke: new ol.style.Stroke({
            color: '#ffcc33',
            width: 2
        }),
        image: new ol.style.Circle({
            radius: 7,
            fill: new ol.style.Fill({
                color: '#ffcc33'
            })
        })
    })
});

map.addLayer(vectorLayer);

// Variables for drawing interactions
var sketch;
var helpTooltipElement;
var helpTooltip;
var measureTooltipElement;
var measureTooltip;
var continuePolygonMsg = 'Click to continue drawing the polygon';
var continueLineMsg = 'Click to continue drawing the line';
var draw; 

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

var formatLength = function(line) {
    var length = ol.sphere.getLength(line, {
        projection: 'EPSG:4326'
    });
    //var length = getLength(line);
    //var length = line.getLength({projection:'EPSG:4326'});

    var output;
    if (length > 100) {
        output = (Math.round(length / 1000 * 100) / 100) +
            ' ' + 'km';

    } else {
        output = (Math.round(length * 100) / 100) +
            ' ' + 'm';

    }
    return output;

};


/**
 * Format area output.
 * @param {module:ol/geom/Polygon~Polygon} polygon The polygon.
 * @return {string}// Formatted area.
 */
var formatArea = function(polygon) {
    // var area = getArea(polygon);
    var area = ol.sphere.getArea(polygon, {
        projection: 'EPSG:4326'
    });
    // var area = polygon.getArea();
    //alert(area);
    var output;
    if (area > 10000) {
        output = (Math.round(area / 1000000 * 100) / 100) +
            ' ' + 'km<sup>2</sup>';
    } else {
        output = (Math.round(area * 100) / 100) +
            ' ' + 'm<sup>2</sup>';
    }
    return output;
};

function addInteraction() {
    if (measuretype.value == 'select' || measuretype.value == 'clear') {

        if (draw) {
            map.removeInteraction(draw)
        };
        if (vectorLayer) {
            vectorLayer.getSource().clear();
        }
        if (helpTooltip) {
            map.removeOverlay(helpTooltip);
        }

        if (measureTooltipElement) {
            var elem = document.getElementsByClassName("tooltip tooltip-static");
            //$('#measure_tool').empty(); 

            //alert(elem.length);
            for (var i = elem.length - 1; i >= 0; i--) {

                elem[i].remove();
                //alert(elem[i].innerHTML);
            }
        }

        //var elem1 = elem[0].innerHTML;
        //alert(elem1);

    } else if (measuretype.value == 'area' || measuretype.value == 'length') {
        var type;
        if (measuretype.value == 'area') {
            type = 'Polygon';
        } else if (measuretype.value == 'length') {
            type = 'LineString';
        }
        //alert(type);

        //var type = (measuretype.value == 'area' ? 'Polygon' : 'LineString');
        draw = new ol.interaction.Draw({
            source: source,
            type: type,
            style: new ol.style.Style({
                fill: new ol.style.Fill({
                    color: 'rgba(255, 255, 255, 0.5)'
                }),
                stroke: new ol.style.Stroke({
                    color: 'rgba(0, 0, 0, 0.5)',
                    lineDash: [10, 10],
                    width: 2
                }),
                image: new ol.style.Circle({
                    radius: 5,
                    stroke: new ol.style.Stroke({
                        color: 'rgba(0, 0, 0, 0.7)'
                    }),
                    fill: new ol.style.Fill({
                        color: 'rgba(255, 255, 255, 0.5)'
                    })
                })
            })
        });
        map.addInteraction(draw);
        createMeasureTooltip();
        createHelpTooltip();
        /**
         * Handle pointer move.
         * @param {module:ol/MapBrowserEvent~MapBrowserEvent} evt The event.
         */
        var pointerMoveHandler = function(evt) {
            if (evt.dragging) {
                return;
            }
            /** @type {string} */
            var helpMsg = 'Click to start drawing';

            if (sketch) {
                var geom = (sketch.getGeometry());
                if (geom instanceof ol.geom.Polygon) {

                    helpMsg = continuePolygonMsg;
                } else if (geom instanceof ol.geom.LineString) {
                    helpMsg = continueLineMsg;
                }
            }

            helpTooltipElement.innerHTML = helpMsg;
            helpTooltip.setPosition(evt.coordinate);

            helpTooltipElement.classList.remove('hidden');
        };

        map.on('pointermove', pointerMoveHandler);

        map.getViewport().addEventListener('mouseout', function() {
            helpTooltipElement.classList.add('hidden');
        });


        var listener;
        draw.on('drawstart',
            function(evt) {
                // set sketch


                //vectorLayer.getSource().clear();

                sketch = evt.feature;

                /** @type {module:ol/coordinate~Coordinate|undefined} */
                var tooltipCoord = evt.coordinate;

                listener = sketch.getGeometry().on('change', function(evt) {
                    var geom = evt.target;

                    var output;
                    if (geom instanceof ol.geom.Polygon) {

                        output = formatArea(geom);
                        tooltipCoord = geom.getInteriorPoint().getCoordinates();

                    } else if (geom instanceof ol.geom.LineString) {

                        output = formatLength(geom);
                        tooltipCoord = geom.getLastCoordinate();
                    }
                    measureTooltipElement.innerHTML = output;
                    measureTooltip.setPosition(tooltipCoord);
                });
            }, this);

        draw.on('drawend',
            function() {
                measureTooltipElement.className = 'tooltip tooltip-static';
                measureTooltip.setOffset([0, -7]);
                // unset sketch
                sketch = null;
                // unset tooltip so that a new one can be created
                measureTooltipElement = null;
                createMeasureTooltip();
                ol.Observable.unByKey(listener);
            }, this);

    }
}


/**
 * Creates a new help tooltip
 */
function createHelpTooltip() {
    if (helpTooltipElement) {
        helpTooltipElement.parentNode.removeChild(helpTooltipElement);
    }
    helpTooltipElement = document.createElement('div');
    helpTooltipElement.className = 'tooltip hidden';
    helpTooltip = new ol.Overlay({
        element: helpTooltipElement,
        offset: [15, 0],
        positioning: 'center-left'
    });
    map.addOverlay(helpTooltip);
}


/**
 * Creates a new measure tooltip
 */
function createMeasureTooltip() {
    if (measureTooltipElement) {
        measureTooltipElement.parentNode.removeChild(measureTooltipElement);
    }
    measureTooltipElement = document.createElement('div');
    measureTooltipElement.className = 'tooltip tooltip-measure';

    measureTooltip = new ol.Overlay({
        element: measureTooltipElement,
        offset: [0, -15],
        positioning: 'bottom-center'
    });
    map.addOverlay(measureTooltip);

}