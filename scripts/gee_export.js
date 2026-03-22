/***************************************
 Daily tabular dataset for rainfall forecasting
 Google Earth Engine JavaScript
****************************************/

// ================================
// 1) STUDY AREA
// ===============================
// Ensure your studyArea is defined before this runs
Map.centerObject(studyArea, 9);
Map.addLayer(studyArea, {color: 'red'}, 'Study Area');

// ================================
// 2) DATE RANGE
// ================================
var startDate = '2001-01-01';
var endDate   = '2026-02-28';

// ================================
// 3) DATASETS & PREP
// ================================
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(startDate, endDate)
  .filterBounds(studyArea)
  .select(['precipitation'], ['rain_mm']);

var era5land = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate(startDate, endDate)
  .filterBounds(studyArea);

var era5Prepared = era5land.map(function(img) {
  var tempC = img.select('temperature_2m').subtract(273.15).rename('temp_c');
  var dewpointC = img.select('dewpoint_temperature_2m').subtract(273.15).rename('dewpoint_c');
  var pressureHpa = img.select('surface_pressure').divide(100.0).rename('surface_pressure_hpa');
  var u10 = img.select('u_component_of_wind_10m').rename('u10_ms');
  var v10 = img.select('v_component_of_wind_10m').rename('v10_ms');
  var windSpeed = u10.pow(2).add(v10.pow(2)).sqrt().rename('wind_speed_ms');
  var soilWater = img.select('volumetric_soil_water_layer_1').rename('soil_water_l1');
  var solarRad = img.select('surface_net_solar_radiation_sum').divide(1e6).rename('solar_rad_MJm2');
  var tpMm = img.select('total_precipitation_sum').multiply(1000.0).rename('era5land_tp_mm');

  return ee.Image.cat([
    tempC, dewpointC, pressureHpa, u10, v10, windSpeed, soilWater, solarRad, tpMm
  ]).copyProperties(img, ['system:time_start']);
});

// ================================
// 4) JOIN COLLECTIONS BY DATE
// ================================
var join = ee.Join.inner();
var filter = ee.Filter.equals({
  leftField: 'system:time_start',
  rightField: 'system:time_start'
});

// apply() directly outputs an ee.FeatureCollection
var joinedFeatures = join.apply(chirps, era5Prepared, filter);

// ================================
// 5) SPATIAL REDUCTION & TABLE BUILD
// ================================
// Map directly over the FeatureCollection to avoid ImageCollection casting bugs
var dailyTable = joinedFeatures.map(function(feature) {
  // Extract matched images from the join
  var primary = ee.Image(feature.get('primary'));
  var secondary = ee.Image(feature.get('secondary'));
  
  // Combine into one image
  var mergedImg = primary.addBands(secondary);

  // Reduce region
  var stats = mergedImg.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea,
    scale: 5566,
    bestEffort: true,
    maxPixels: 1e13
  });

  // Define default values. If reduceRegion returns missing data for a day,
  // this ensures the column structure remains perfectly consistent.
  var defaultStats = ee.Dictionary({
    'rain_mm': null,
    'temp_c': null,
    'dewpoint_c': null,
    'surface_pressure_hpa': null,
    'u10_ms': null,
    'v10_ms': null,
    'wind_speed_ms': null,
    'soil_water_l1': null,
    'solar_rad_MJm2': null,
    'era5land_tp_mm': null
  });
  
  var finalStats = defaultStats.combine(stats);

  // Extract time
  var timestamp = primary.get('system:time_start');
  var dateObj = ee.Date(timestamp);
  var month = dateObj.get('month');
  var doy = ee.Number(dateObj.getRelative('day', 'year')).add(1);
  var angle = doy.multiply(2 * Math.PI).divide(365.0);

  // Combine temporal features with statistics using ee.Dictionary.combine
  var props = ee.Dictionary({
    'date': dateObj.format('YYYY-MM-dd'),
    'month': month,
    'day_of_year': doy,
    'doy_sin': angle.sin(),
    'doy_cos': angle.cos(),
    'system:time_start': timestamp // Keeps the native time object for .filterDate()
  }).combine(finalStats);

  return ee.Feature(null, props);
});

// ================================
// 6) PREVIEW & CHART
// ================================
print('Daily table preview (first 10 rows):', dailyTable.limit(10));

// Filter to one year to prevent browser crash during charting
var chartPreviewCol = dailyTable.filterDate(startDate, '2002-01-01');

var chart = ui.Chart.feature.byFeature({
  features: chartPreviewCol,
  xProperty: 'date',
  yProperties: ['rain_mm']
}).setOptions({
  title: 'Daily CHIRPS Rainfall Preview (2001 Only)',
  hAxis: {title: 'Date'},
  vAxis: {title: 'Rainfall (mm/day)'},
  lineWidth: 1,
  pointSize: 0
});

print(chart);

// ================================
// 7) EXPORT CSV TO GOOGLE DRIVE
// ================================
Export.table.toDrive({
  collection: dailyTable,
  description: 'Rainfall_TimeSeries_Tabular_Dataset',
  fileFormat: 'CSV'
});