var roi = ee.Geometry.Rectangle([
  -89.40, 13.708,
  -89.26, 13.85
])


Map.centerObject(roi, 13);
Map.addLayer(roi, {color: 'red'}, 'Área de Interés', false);

var dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filterDate('2025-01-01', '2025-10-30') 
                  // Filtro estricto de nubes: menos del 10%
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10));


var imagenLimpia = dataset.median().clip(roi);


var visParams = {
  min: 0,
  max: 3000,
  bands: ['B4', 'B3', 'B2'],
};
Map.addLayer(imagenLimpia, visParams, 'Sentinel-2 (Color Verdadero)');

var bandasParaExportar = imagenLimpia.select(['B4', 'B3', 'B2', 'B8', 'B11']);

Export.image.toDrive({
  image: bandasParaExportar,
  description: 'Sentinel2Zaragoza_2025',
  folder: 'Tesis_Imagenes_real', 
  scale: 10, // Resolución nativa de 10 metros por píxel
  region: roi,
  crs: 'EPSG:32616',
  maxPixels: 1e9 
});