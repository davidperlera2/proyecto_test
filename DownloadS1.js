var roi = ee.Geometry.Rectangle([
  -89.40, 13.708,
  -89.26, 13.85
])

function obtenerImagenSAR(año) {
  var fechaInicio = año + '-01-01';
  var fechaFin = año + '-11-30'; 
  
  var datasetSAR = ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(roi)
                  .filterDate(fechaInicio, fechaFin)
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
                  
  var s1 = datasetSAR.median().clip(roi).select(['VV', 'VH']);
  
  
  var s1_filtrado = s1.focal_median(20, 'circle', 'meters');



  // Ratio VV/VH 
  var ratio = s1_filtrado.select('VV')
                   .divide(s1_filtrado.select('VH'))
                   .rename('VV_VH_ratio');

  var s1_final = s1_filtrado.addBands(ratio).reproject({
    crs: 'EPSG:32616',
    scale: 10
  });

  return s1_final;
}


var sar2025 = obtenerImagenSAR('2025');

Map.addLayer(sar2025, {min: -25, max: 0, bands: ['VV']}, 'SAR VV (2025)', false);



Export.image.toDrive({
  image: sar2025,
  description: 'SAR_2017',
  folder: 'Tesis_Imagenes_real',
  scale: 10,
  region: roi,
  crs: 'EPSG:32616',
  maxPixels: 1e9
});