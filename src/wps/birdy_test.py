from birdy import WPSClient
from pathlib import Path

pywps = WPSClient('http://localhost:5000/wps')

cost_raster = Path(r"./../../results/weights/result_res_100_all_touched_True.tif")
start_features = Path(r"../../results/test_points/start_point.gpkg")
end_features = Path(r"../../results/test_points/end_point.gpkg")

print(pywps.lcp.__doc__)
result = pywps.lcp(costs=cost_raster,
                   start=start_features,
                   end=end_features)
print(result.get(asobj=True)[0])  # print the geojson?
