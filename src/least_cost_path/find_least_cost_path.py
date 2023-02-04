from math import floor

import numpy as np
from geopandas import read_file, GeoSeries, GeoDataFrame
from pandas import concat
from pathlib import Path
from rioxarray import open_rasterio
from shapely.geometry import Point, LineString, MultiPoint, box
from xarray import DataArray

from src.least_cost_path.dijkstra import dijkstra, backtrack_path
from src.rules2weights import write_compressed
from argparse import ArgumentParser


# TODO comment for function descriptions
def _point_to_row_col(point_xy: Point, raster_layer: DataArray) -> tuple[int, int]:
    """
    Convert a point from Shapely Point, to index coordinates of the point in `raster_layer`
    :param point_xy: Shapely Point
    :param raster_layer: Raster as DataArray
    :return: tuple[int, in]
    """
    affine = raster_layer.rio.transform()
    x_res = affine.a
    y_res = abs(affine.e)
    x_min, _, _, y_max = raster_layer.rio.bounds()

    col = floor((point_xy.x - x_min) / x_res)
    row = floor((y_max - point_xy.y) / y_res)

    return row, col


def features_to_tuples(point_features: GeoSeries, raster_layer: DataArray) -> list[tuple[tuple[int, int], Point, int]]:
    """
    Convert all Points in `point_features` into a list of the points as tuple-coordinates, original Coords, and Index
    :param point_features: GeoSeries of Shapley Points
    :param raster_layer: DataArray
    :return: list[tuple[tuple[int, int], Point, int]] list of tuple of Point coord as tuple, Shapely Point and  index
    """
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


def _row_col_to_point(row_col: tuple[int], raster_layer: DataArray) -> Point:
    """
    Convert tuple coordinates into Shapely Points (with correct offset)
    :param row_col: tuple[int, int], relative coordinates
    :param raster_layer: DataArray
    :return: Point
    """
    affine = raster_layer.rio.transform()
    x_res = affine.a
    y_res = abs(affine.e)
    x_min, y_min, _, y_max = raster_layer.rio.bounds()

    x = (row_col[1] + 0.5) * x_res + x_min
    y = y_max - (row_col[0] + 0.5) * y_res
    return Point(x, y)


def raster2matrix(block: DataArray) -> tuple[list[list[None | float]], bool]:
    """
    Convert the raster to a 2D-List
    :param block: DataArray
    :return: tuple[list[list[None | float]], bool]
    """
    _contains_negative = False
    _matrix = [[None if block.data[i, j] == block.rio.nodata else block.data[i, j] for j in range(block.rio.width)]
               for i in range(block.rio.height)]

    for line in _matrix:
        for value in line:
            if value is not None:
                if value < 0:
                    _contains_negative = True

    return _matrix, _contains_negative


def matrix2raster(matrix: dict, alike_raster: DataArray, nodata: None | float = None) -> DataArray:
    """
    Convert the 2D-List back into the original raster format
    :param matrix: list[list[None | float]] the 2D-List to convert
    :param alike_raster: the raster with the correct crs, shape, nodata
    :param nodata: if given: value to use as nodata
    :return: DataArray
    """
    return_raster = alike_raster.copy()
    return_raster = return_raster.rio.write_nodata(nodata)
    return_raster.data = np.full_like(return_raster.data, fill_value=return_raster.rio.nodata)

    for i in range(alike_raster.rio.height):
        for j in range(alike_raster.rio.width):
            if (i, j) in matrix:
                return_raster[i, j] = matrix[(i, j)]
    return return_raster


def create_points_from_path(_cost_raster: DataArray, min_cost_path: list[tuple[int, int]], start_point: Point,
                            end_point: Point) -> list[Point]:
    """
    Convert Path form 2D-List format to a list of Shapley Points
    :param _cost_raster: DataArray
    :param min_cost_path: list[tuple[int, int]]
    :param start_point: Point
    :param end_point: Point
    :return: list[Point]
    """
    path_points = list(map(lambda row_col: _row_col_to_point(row_col, _cost_raster), min_cost_path))
    path_points[0] = start_point
    path_points[-1] = end_point

    return path_points


def create_path_feature_from_points(_path_points: list[Point], attr_vals: tuple[int, int, float]) -> GeoDataFrame:
    """
    Convert Path form 2D-List format to a GeoDataFrame containing the path as Shapley LineString and MetaData
    from attr_vals
    :param _path_points: list[Point]
    :param attr_vals: tuple[int, int, float]
    :return: GeoDataFrame
    """
    polyline = LineString(_path_points)
    feature = GeoDataFrame(data={'geometry': [polyline, ],
                                 'start point id': [attr_vals[0], ],
                                 'end point id': [attr_vals[1], ],
                                 'total cost': attr_vals[2], }
                           )

    return feature


def find_least_cost_path(cost_raster: DataArray, cost_raster_band: int, is_nearest: bool, start_features: GeoDataFrame,
                         end_features: GeoDataFrame, aggregated_save: None | str = None) -> GeoDataFrame:
    """
    Compute the Least Cost Path with Dijkstra Algorithm
    :param cost_raster: DataArray
    :param cost_raster_band: int
    :param is_nearest: bool
    :param start_features: GeoDataFrame of the single starting POINT
    :param end_features: GeoDataFrame of at least one END POINTS
    :param aggregated_save: optional str: path to save the aggregate raster, if given
    :return: GeoDataFrame
    """
    raster_2d = cost_raster[cost_raster_band]

    start_tuples = features_to_tuples(start_features.geometry, cost_raster)
    end_tuples = features_to_tuples(end_features.geometry, cost_raster)
    matrix, contains_negative = raster2matrix(raster_2d)
    if contains_negative:
        return GeoDataFrame()

    _path_features = []
    for itr, (_map, _indexes,  _end_node) in enumerate(dijkstra(start_tuples[0], end_tuples, matrix, is_nearest)):
        (path, costs), terminal_tuple = *backtrack_path(_map, _indexes, _end_node, start_tuples[0]), end_tuples[itr]
        if aggregated_save:
            aggregated_raster = matrix2raster(_map, raster_2d, -9999)
            write_compressed(aggregated_raster, Path(aggregated_save))

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
    parser.add_argument('save_name', type=str, help='Path where this projects vector file can the saved')
    parser.add_argument('-aggregated_name', type=str, help='Path where this aggregated raster file can the saved')

    args = parser.parse_args()

    path_features = find_least_cost_path(open_rasterio(args.cost_raster), args.cost_raster_band, False,
                                         read_file(args.start_features), read_file(args.end_features),
                                         args.aggregated_name)
    path_features.to_file(args.save_name)
