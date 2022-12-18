from birdy import WPSClient
from pathlib import Path

pywps = WPSClient('http://localhost:5000/wps')
line_path = Path(r'.\..\..\results\least_cost_paths\least_cost_path_test_points_res_10_al_false.gml')
print(line_path.exists())
line = open(line_path)

cost_raster = Path(r".\..\..\results\weights\result_res_100_all_touched_True.tif")
start_features = Path(r"..\..\results\test_points\start_point.gpkg")
end_features = Path(r"..\..\results\test_points\end_point.gpkg")

pywps.lcp(costs=cost_raster,
          start=start_features,
          end=end_features).get(asobj=True)[0]

# pywps.buffer(poly_in=line,
#              buffer=10.0).get(asobj=True)[0]
