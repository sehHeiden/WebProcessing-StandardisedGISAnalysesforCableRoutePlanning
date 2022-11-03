from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from pprint import pprint

import fiona
import numpy as np
from brackettree import Node
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from geopandas import read_file, GeoDataFrame
from pandas import read_excel, DataFrame, concat, merge
from rioxarray import open_rasterio
from rioxarray.merge import merge_arrays


def write_compress(r, save: Path) -> None:
    predict = 3 if r.dtype == 'float' else 2
    r.rio.to_raster(save, tiled=True, compress='ZSTD', num_threads='ALL_CPUS', predictor=predict, zstd_level=1)


def filter_gdf(_gdf: GeoDataFrame, columns: list, values: list):
    # STARTSWITH
    is_negative = False
    new_values = []
    for value in values:
        is_negative = True if 'NOT' in value else False  # TODO will cause troubles
        value = value.replace('NOT ', '')
        if 'STARTSWITH(' in value:
            value = value[11:-1]
            if ' OR ' in value:
                new_values.append(value.split[' OR '])
            else:
                new_values.append([value])
            values = [x1 for x in new_values for x1 in x]
    # TODO replace new_values with old, but only when necessary

    if len(columns) == len(values):
        return concat([_gdf[is_negative * _gdf[column] == value]] for column, value in zip(columns, values))
    elif len(columns) == 1 and len(values) > 1:
        return _gdf[is_negative * _gdf[columns[0]].isin(values)]


def buffer(_gdf: GeoDataFrame, distance: float | str) -> GeoDataFrame:
    _gdf = _gdf.copy()

    if isinstance(distance, str) and 'COL:' in distance:
        buffer_start = distance.find('COL:')
        col_name = distance[buffer_start + 4:].split(' ', 1)[0]
        buffer_end = buffer_start + len(col_name)
        buffer_value = val if ~ (val := _gdf[col_name]).isna() else 0

        distance = distance[:buffer_start] + str(buffer_value) + distance[buffer_end:]

    if isinstance(distance, str) and distance.startswith('MAX'):
        distance = distance[4:-1]
        distance = [float(x) for x in distance.split(' OR ')]
        distance = np.max(distance)

    _gdf['geometry'] = _gdf['geometry'].buffer(distance if not np.isnan(distance) else 0)
    return _gdf


def _split_col(split_str: str | float | int):
    if isinstance(split_str, str):
        if ' OR ' in split_str and ' OR ' in Node(split_str).items[0]:  # if OR and it's outside the brackets

            split_str = split_str.split(' OR ')
            if np.all([x.isdigit() for x in split_str]):
                split_str = [float(x) for x in split_str]
            return split_str
        else:
            return [split_str]
    elif isinstance(split_str, float) and np.isnan(split_str):
        return []

    else:
        return [split_str]


def make_processable(_df: DataFrame, main_path: Path) -> DataFrame:
    _df = _df.copy()

    # drop Lines not usable (Use == NO), or lines without a path (dir AND FileName)
    _df = _df.loc[~_df['Directory'].isna() | ~_df['FileName'].isna()]
    _df = _df.loc[~(_df['Use'] == 'No')]
    assert np.any(_df['Use'] == 'Yes')

    # SPLIT OR into list
    for col in _df.columns:
        _df[col] = _df[col].apply(lambda x: _split_col(x))

    _df['Path'] = _df.apply(lambda x: (main_path / x.Directory[0]) / x.FileName[0], axis=1)
    _df['Level'] = _df['Level'].apply(lambda x: x[0])
    _df['Description'] = _df['Description'].apply(lambda x: x[0])
    _df['LineNum'] = (_df.index + 2)

    return _df.loc[:, ['LineNum', 'Path', 'Description', 'Layer', 'Level', 'Column', 'ColumnValue', 'Buffer']]


def process_default_rule(df: DataFrame, resolution: float | tuple[float, list], crs: int,
                         _save_dir: Path, _all_touched: bool) -> list[str | Path]:
    named_tuple = list((df.copy().loc[df['Description'] == 'Base']).itertuples())[0]
    return process_rule(named_tuple, resolution, crs, _save_dir, _all_touched)


def process_rule(named_tuple, resolution: float | tuple[float, list], crs: int,
                 _save_dir: Path, _all_touched: bool) -> list[str | Path]:
    gdf = read_file(named_tuple.Path) if len(named_tuple.Layer) == 0 else read_file(named_tuple.Path,
                                                                                    layer=named_tuple.Layer[0])
    gdf = gdf.to_crs(epsg=crs)

    if len(named_tuple.Column) > 0:
        gdf = filter_gdf(gdf, named_tuple.Column, named_tuple.ColumnValue)
    gdf = buffer(gdf, named_tuple.Buffer)

    save_path = _save_dir / named_tuple.Path
    raster = make_geocube(gdf, resolution=resolution, rasterize_function=partial(rasterize_image, _all_touched))
    write_compress(raster, save_path)
    return save_path


if __name__ == '__main__':
    parser = ArgumentParser(description='Read in Vectors from Configuration file and rasterize them.')
    parser.add_argument('config_file_path', type=str, help='Full path to the .xlsx config file.')
    parser.add_argument('vectors_main_folder', type=str, help='Main Path to the vector files.')
    parser.add_argument('save_dir', type=str, help='Path where this projects raster files can the saved')
    parser.add_argument('resolution', type=int, default=1000, help='Resolution to use.')
    parser.add_argument('all_touched', type=bool, default=False, help='Select pixel only that`s pixel center  is '
                                                                      'covered by the Polygone (False) or any point'
                                                                      ' of the pixel overlaps with the Polygon (True).')
    parser.add_argument('crs', type=int, default=3157, help='Reference system. Default: Google Pseudo Mercator.')

    args = parser.parse_args()

    pprint(fiona.supported_drivers)
    rules = read_excel(args.config_file_path, sheet_name='ProcessingRules')
    weights = read_excel(args.config_file_path, sheet_name='Weights')
    rules = make_processable(rules, Path(args.vectors_main_folder))
    rules = merge(rules, weights, on='Level').drop(columns='Level')

    save_dir = Path(args.save_dir)
    default_save = process_default_rule(rules, args.resolution, args.crs, save_dir, args.all_touched)
    rules = rules.loc[~rules['Description'] == 'Basis']

    processed_rules = []
    for rule in rules.iterrows():
        processed_rules = processed_rules + process_rule(rule, args.resolution, args.crs, save_dir, args.all_touched)

    # combine the rasters file
    # processed_rules -> max, processed_rules + default (where processed is nodata)
    processed_rules = merge_arrays([open_rasterio(x, cache=False) for x in processed_rules], method='max')
    default_rules = open_rasterio(default_save)

    processed_rules.data = np.where(processed_rules == processed_rules.rio.nodata, default_rules, processed_rules)
    write_compress(processed_rules, save_dir / 'result.tif')
