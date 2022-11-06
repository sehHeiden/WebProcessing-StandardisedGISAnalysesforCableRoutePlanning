from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from pprint import pprint

import fiona
import numpy as np
from brackettree import Node
from brackettree.nodes import TextNode, RoundNode
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from geopandas import read_file, GeoDataFrame
from pandas import read_excel, DataFrame, concat, merge, Series
from rioxarray import open_rasterio
from rioxarray.merge import merge_arrays
from tqdm import tqdm
from xarray import DataArray


def write_compress(r: DataArray, save: Path) -> None:
    predict = 3 if np.issubdtype(r.dtype, np.floating) else 2
    r.rio.to_raster(save, driver='GTiff', tiled=True, compress='ZSTD',
                    num_threads='ALL_CPUS', predictor=predict, zstd_level=1)


def __filter(total_gdf: GeoDataFrame, column: str, _rule: Node, ) -> GeoDataFrame:
    child_results: list[GeoDataFrame] = []
    for item in _rule.items:
        if isinstance(item, RoundNode):
            child_results.append(__filter(total_gdf, column, item))

    if len(child_results) > 1:
        child_result = concat(child_results)
    elif len(child_results) == 1:
        child_result = child_results[0]
    else:
        child_result = GeoDataFrame(data=[], columns=total_gdf.columns, crs=total_gdf.crs)

    operator, identifiers = get_command(_rule)

    if operator == 'NOT':
        if len(identifiers) == 0:
            return total_gdf[~total_gdf.index.isin(child_result.index)]
        else:
            return total_gdf[~(total_gdf[column] == identifiers[0])]

    elif operator == 'OR':
        if len(identifiers) > 0:
            new_results = total_gdf[total_gdf[column].isin(identifiers)]
            return concat([child_result, new_results])
        else:
            return child_result

    elif operator == 'STARTSWITH':
        return total_gdf[total_gdf[column].apply(lambda x: isinstance(x, str) and x.startswith(identifiers[0]))]
    else:
        return child_result


def filter_gdf(_gdf: GeoDataFrame, column: str, values: str):
    # decompose the value
    start_node = Node(values)
    return __filter(_gdf, column, start_node)


def buffer(_gdf: GeoDataFrame, distance: float | str) -> GeoDataFrame:
    _gdf = _gdf.copy()

    if isinstance(distance, str):
        distance = __estimate_buffer_size(_gdf, Series(np.zeros(len(_gdf)), index=_gdf.index), Node(distance))
    else:
        distance = distance if not np.isnan(distance) else 0

    _gdf['geometry'] = _gdf['geometry'].buffer(distance)
    return _gdf


def __estimate_buffer_size(data: GeoDataFrame, _buffer: Series, _rule: Node, ) -> Series:
    child_results: list[Series | np.ndarray] = []
    for item in _rule.items:
        if isinstance(item, RoundNode):
            child_results.append(__estimate_buffer_size(data, _buffer, item))

    operator, identifiers = get_command(_rule)
    if operator == 'MAX':
        return Series(np.max([*[x.values for x in child_results],
                              *[np.full(len(data), float(x)) for x in identifiers if x.isdigit()]],
                             axis=0),
                      index=data.index)
    elif operator == 'COL':
        return data[identifiers[0]].apply(lambda x: x if isinstance(x, float) and ~ np.isnan(x) and x != -9998 else 0)
    elif operator == 'RoundNode':
        return concat(child_results)
    else:
        raise NotImplementedError


def get_command(_rule):
    command = str(_rule.items[0]).split(' ')
    command = [x.strip() for x in command]
    operator = command[0]
    identifiers = [x for x in command[1:] if len(x) > 0]
    for i in _rule.items[1:]:
        if isinstance(i, TextNode):
            new_identifier = str(i).strip()
            if len(new_identifier) > 0:
                identifiers.append(new_identifier)
    return operator, identifiers


def make_processable(_df: DataFrame, main_path: Path) -> DataFrame:
    _df = _df.copy()

    # drop Lines not usable (Use == NO), or lines without a path (dir AND FileName)
    _df = _df.loc[~_df['Directory'].isna() | ~_df['FileName'].isna()]
    _df = _df.loc[~(_df['Use'] == 'No')]
    assert np.any(_df['Use'] == 'Yes')

    _df['Path'] = _df.apply(lambda x: (main_path / x.Directory) / x.FileName, axis=1)
    _df['LineNum'] = (_df.index + 2)

    return _df.loc[:, ['LineNum', 'Path', 'Description', 'Layer', 'Level', 'Column', 'ColumnValue', 'Buffer']]


def __vector_processing(named_tuple, _crs: int, base_extent: tuple | None = None) -> GeoDataFrame:
    layer_name = named_tuple.Layer if isinstance(named_tuple.Layer, str) else None
    gdf: GeoDataFrame = read_file(named_tuple.Path, bbox=base_extent, layer=layer_name)
    gdf = gdf.to_crs(epsg=_crs)
    gdf['Weights'] = named_tuple.Weight

    if isinstance(named_tuple.Column, str):
        gdf = filter_gdf(gdf, named_tuple.Column, named_tuple.ColumnValue)
    return buffer(gdf, named_tuple.Buffer)


def process_base_rule(df: DataFrame, _resolution: float | tuple[float, list], _crs: int,
                      _save_dir: Path, _all_touched: bool) -> DataArray:
    named_tuple = list((df.copy().loc[df['Description'] == 'Base']).itertuples())[0]
    vec = __vector_processing(named_tuple, _crs)
    save_path = _save_dir / f"rule_{named_tuple.LineNum}_{named_tuple.Description}.tif"
    raster = make_geocube(vec, ['Weights', ], resolution=resolution, fill=999,
                          rasterize_function=partial(rasterize_image, all_touched=_all_touched))
    raster = raster['Weights']
    write_compress(raster, save_path)
    return raster


def process_rule(named_tuple, example_da: DataArray, _crs: int,
                 _save_dir: Path, _all_touched: bool, bbox) -> list[str | Path]:
    vec = __vector_processing(named_tuple, _crs, bbox)

    save_path = _save_dir / f"rule_{named_tuple.LineNum}_{named_tuple.Description}.tif"
    raster = make_geocube(vec, ['Weights'], like=example_da, fill=999,
                          rasterize_function=partial(rasterize_image, all_touched=_all_touched))
    raster = raster['Weights']
    raster.data[example_da == example_da.rio.nodata] = raster.rio.nodata
    write_compress(raster, save_path)
    return [save_path, ]


if __name__ == '__main__':
    parser = ArgumentParser(description='Read in Vectors from Configuration file and rasterize them.')
    parser.add_argument('config_file_path', type=str, help='Full path to the .xlsx config file.')
    parser.add_argument('vectors_main_folder', type=str, help='Main Path to the vector files.')
    parser.add_argument('save_dir', type=str, help='Path where this projects raster files can the saved')
    parser.add_argument('-r', '--resolution', type=int, default=1000, help='Resolution to use.')
    parser.add_argument('-at', '--all_touched', action='store_true',  help='Select those pixels, those CENTER  is '
                                                                           'covered by the Polygone (False) or any '
                                                                           'part overlaps with the Polygone (True).')
    parser.add_argument('--crs', type=int, default=3157, help='Reference system. Default: Google Pseudo Mercator.')

    args = parser.parse_args()

    pprint(fiona.supported_drivers)
    rules = read_excel(args.config_file_path, sheet_name='ProcessingRules')
    weights = read_excel(args.config_file_path, sheet_name='Weights')
    rules = make_processable(rules, Path(args.vectors_main_folder))
    rules = merge(rules, weights, on='Level').drop(columns='Level')
    resolution, crs, all_touched = args.resolution, args.crs, args.all_touched

    save_dir = Path(args.save_dir)
    base_raster = process_base_rule(rules, resolution, crs, save_dir, all_touched)
    rules = rules.loc[~(rules['Description'] == 'Base')]

    finalized_rules = []
    p_bar = tqdm(rules.itertuples(), total=len(rules))
    for rule in p_bar:
        p_bar.set_description(f"Rule {rule.LineNum}: {rule.Description}.")
        finalized_rules = finalized_rules + process_rule(rule, base_raster, crs, save_dir, all_touched,
                                                         base_raster.rio.bounds())

    # combine the rasters file
    # finalized_rules -> max, finalized_rules + default (where processed is nodata)
    finalized_rules = merge_arrays([open_rasterio(x, cache=False) for x in finalized_rules], method='max')

    finalized_rules.data = np.where(finalized_rules == finalized_rules.rio.nodata, base_raster, finalized_rules)
    write_compress(finalized_rules, save_dir / f'result_res_{resolution}_all_touched_{all_touched}.tif')
