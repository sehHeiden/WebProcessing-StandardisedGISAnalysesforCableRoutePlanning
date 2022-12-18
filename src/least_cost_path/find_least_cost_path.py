from math import floor

from geopandas import read_file, GeoSeries, GeoDataFrame
from pandas import concat
from rioxarray import open_rasterio
from shapely.geometry import Point, LineString, MultiPoint, box
from xarray import DataArray

from src.least_cost_path.dijkstra import dijkstra
from argparse import ArgumentParser


# TODO comment for function descriptions
def _point_to_row_col(point_xy: Point, raster_layer: DataArray):
    affine = raster_layer.rio.transform()
    x_res = affine.a
    y_res = abs(affine.e)
    x_min, _, _, y_max = raster_layer.rio.bounds()

    col = floor((point_xy.x - x_min) / x_res)
    row = floor((y_max - point_xy.y) / y_res)

    return row, col


def features_to_tuples(point_features: GeoSeries, raster_layer: DataArray) -> list[tuple]:
    # TODO check data type
    row_cols = []

    extent = box(*raster_layer.rio.bounds())

    for idx, point_feature in enumerate(point_features):
        if point_feature.is_valid:

            if isinstance(point_feature, MultiPoint):
                for point_xy in point_feature:
                    if extent.contains(point_xy):
                        row_col = _point_to_row_col(point_xy, raster_layer)
                        row_cols.append((row_col, point_xy, idx))

            elif isinstance(point_feature, Point):
                if extent.contains(point_feature):
                    row_col = _point_to_row_col(point_feature, raster_layer)
                    row_cols.append((row_col, point_feature, idx))

    return row_cols


def _row_col_to_point(row_col, raster_layer: DataArray):
    affine = raster_layer.rio.transform()
    x_res = affine.a
    y_res = abs(affine.e)
    x_min, y_min, _, y_max = raster_layer.rio.bounds()

    x = (row_col[1] + 0.5) * x_res + x_min
    y = y_max - (row_col[0] + 0.5) * y_res
    return Point(x, y)


def raster2matrix(block: DataArray) -> tuple[list[list[None | float]], bool]:
    _contains_negative = False
    _matrix = [[None if block.data[i, j] == block.rio.nodata else block.data[i, j] for j in range(block.rio.width)]
               for i in range(block.rio.height)]

    for line in _matrix:
        for value in line:
            if value is not None:
                if value < 0:
                    _contains_negative = True

    return _matrix, _contains_negative


def create_points_from_path(_cost_raster, min_cost_path, start_point, end_point):
    path_points = list(map(lambda row_col: _row_col_to_point(row_col, _cost_raster), min_cost_path))
    path_points[0] = start_point
    path_points[-1] = end_point

    return path_points


def create_path_feature_from_points(_path_points: list[Point], attr_vals):
    polyline = LineString(_path_points)
    feature = GeoDataFrame(data={'geometry': [polyline, ],
                                 'start point id': [attr_vals[0], ],
                                 'end point id': [attr_vals[1], ],
                                 'total cost': attr_vals[2], }
                           )

    return feature


def find_least_cost_path(cost_raster: DataArray, cost_raster_band: int, is_nearest: bool, start_features: GeoDataFrame,
                         end_features: GeoDataFrame) -> GeoDataFrame:
    raster_2d = cost_raster[cost_raster_band]

    start_tuples = features_to_tuples(start_features.geometry, cost_raster)
    end_tuples = features_to_tuples(end_features.geometry, cost_raster)
    matrix, contains_negative = raster2matrix(raster_2d)
    result = dijkstra(start_tuples[0], end_tuples, matrix, is_nearest)

    _path_features = []

    for path, costs, terminal_tuples in result:
        for terminal_tuple in terminal_tuples:
            path_points = create_points_from_path(cost_raster,
                                                  path,
                                                  start_tuples[0][1],
                                                  terminal_tuple[1])

            total_cost = costs[-1]

            _path_features.append(create_path_feature_from_points(path_points,
                                                                  (start_tuples[0][2],
                                                                   terminal_tuple[2],
                                                                   total_cost)
                                                                  )
                                  )

    _path_features = concat(_path_features, )
    _path_features = _path_features.set_crs(start_features.crs)
    return _path_features


if __name__ == '__main__':
    parser = ArgumentParser(description='Find the nearest cost path.')
    parser.add_argument('cost_raster', type=str, help='Full path to the .xlsx config file.')
    parser.add_argument('cost_raster_band', type=int, help='Full path to the .xlsx config file.')

    parser.add_argument('start_features', type=str, help='Main Path to the vector files.')
    parser.add_argument('end_features', type=str, help='Path where this projects raster files can the saved')
    parser.add_argument('save_name', type=str, help='Path where this projects raster files can the saved')

    args = parser.parse_args()

    path_features = find_least_cost_path(open_rasterio(args.cost_raster), args.cost_raster_band, False,
                                         read_file(args.start_features), read_file(args.end_features))
    path_features.to_file(args.save_name)
