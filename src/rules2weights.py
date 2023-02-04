from argparse import ArgumentParser
from functools import partial
from os import cpu_count
from pathlib import Path
from pprint import pprint

import fiona
import numpy as np
from brackettree import Node
from brackettree.nodes import TextNode, RoundNode
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from geopandas import read_file, GeoDataFrame, GeoSeries
from pandas import read_excel, DataFrame, concat, merge, Series
from ray.util.multiprocessing import Pool
from rioxarray import open_rasterio
from rioxarray.merge import merge_arrays
from tqdm import tqdm
from xarray import DataArray


def write_compressed(r: DataArray, save: Path) -> None:
    """
    Saving the raster r at save
    :param r: raster as DataArray
    :param save: save Path
    :return: None (write)
    """
    predict = 3 if np.issubdtype(r.dtype, np.floating) else 2
    r.rio.to_raster(save, driver='GTiff', tiled=True, compress='ZSTD',
                    num_threads='ALL_CPUS', predictor=predict, zstd_level=1)


def write_png(r: DataArray, save: Path) -> None:
    """
    saving the raster r at save as png (gray)
    :param r: raser as DataArray
    :param save: save Path
    :return: None (write)
    """
    # TODO: gray to pseudo colour: RdYlGn
    lowest_value = r.data.min()
    r.data = r.data / lowest_value
    r = r.astype('uint16')
    r.rio.to_raster(save, driver='PNG')


def __filter(total_gdf: GeoDataFrame, column: str, _rule: Node, ) -> GeoDataFrame:
    """
    filtering the values of the GeoDataFrames for the attributes in the `column` of the GeoDataFrames,
     that follow the `_rule`
    :param total_gdf: original GeoDataFrame
    :param column: column name (str) from which the attributes will be used
    :param _rule: rule (as Node) that applies for the values in the column of the GeoDataFrame
    :return: GeoDataFrame with the filter applied
    """
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


def filter_gdf(_gdf: GeoDataFrame, column: str, values: str) -> GeoDataFrame:
    """
    filtering the values of the GeoDataFrames for the attributes in the `column` of the GeoDataFrames,
     that follow the rule in `values`
    :param _gdf: GeoDataFrame
    :param column: name of the column (str)
    :param values: str of the rules
    :return: GeoDataFrame
    """
    # decompose the value
    start_node = Node(values)
    return __filter(_gdf, column, start_node)


def buffer(_gdf: GeoDataFrame, distance: float | str) -> GeoDataFrame:
    """
    Buffering the objects in GeoDataFrame by the value in `distance` or the rule in `distance` if it's a string
    :param _gdf: GeoDataFrame
    :param distance: float or str
    :return: GeoDataFrame
    """
    _gdf = _gdf.copy()

    if isinstance(distance, str):
        distance = __estimate_buffer_size(_gdf, Series(np.zeros(len(_gdf)), index=_gdf.index), Node(distance))
        distance = distance.values
    else:
        distance = np.full(len(_gdf), distance if not np.isnan(distance) else 0,)

    if len(_gdf) > 5_000:  # ray multiprocessing the buffer
        geometry_arrays = np.array_split(_gdf['geometry'].values, cpu_count() - 1)
        distances = np.array_split(distance, cpu_count() - 1)

        def single_buffer(geometry_array, distance_array):
            s = GeoSeries(geometry_array)
            return s.buffer(distance_array)
        with Pool(cpu_count()-1) as p:
            geometries = p.starmap(single_buffer, zip(geometry_arrays, distances))
        p.close()
        _gdf['geometry'] = concat(geometries).values
        p.terminate()
    else:
        _gdf['geometry'] = _gdf['geometry'].buffer(distance)
    return _gdf


def __estimate_buffer_size(data: GeoDataFrame, _buffer: Series, _rule: Node, ) -> Series:
    """
    estimate the size of the Buffer for every GeoObject in the GeoDataFrame
    :param data: GeoDataFrame
    :param _buffer: Series with the current buffer distances
    :param _rule: rule to estimate the size of the buffer as a Node
    :return: Series
    """
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
    """
    split the current `_rule` into an operator (faction) and its identifiers (parameters)
    :param _rule: Node or a subtype?
    :return: str, list[str]
    """
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
    """
    change the DataFrame, so that it can be used to further process the rules it provides
    :param _df: DataFrame
    :param main_path: Path of the main folder from the input data
    :return: DataFrame
    """
    _df = _df.copy()

    # drop Lines not usable (Use == NO), or lines without a path (dir AND FileName)
    _df = _df.loc[~_df['Directory'].isna() | ~_df['FileName'].isna()]
    _df = _df.loc[~(_df['Use'] == 'No')]
    assert np.any(_df['Use'] == 'Yes')

    _df['Path'] = _df.apply(lambda x: (main_path / x.Directory) / x.FileName, axis=1)
    _df['LineNum'] = (_df.index + 2)

    return _df.loc[:, ['LineNum', 'Path', 'Description', 'Layer', 'Level', 'Column', 'ColumnValue', 'Buffer']]


def __vector_processing(named_tuple, _crs: int, base_extent: tuple | None = None) -> GeoDataFrame:
    """
    Process the one single line of the rules DataFrame and return the processed GeoDataFrame
    :param named_tuple: NamedTuple
    :param _crs: EPSG-Code as int
    :param base_extent: optional tuple
    :return: GeoDataFrame
    """
    layer_name = named_tuple.Layer if isinstance(named_tuple.Layer, str) else None
    gdf: GeoDataFrame = read_file(named_tuple.Path, layer=layer_name)
    gdf = gdf.to_crs(epsg=_crs)
    if base_extent is not None:  # filter by the
        x_min, y_min, x_max, y_max = base_extent
        gdf = gdf.cx[x_min:x_max, y_min:y_max]
    gdf['Weights'] = named_tuple.Weight

    if isinstance(named_tuple.Column, str):
        gdf = filter_gdf(gdf, named_tuple.Column, named_tuple.ColumnValue)
    return buffer(gdf, named_tuple.Buffer)


def process_base_rule(df: DataFrame, _res: float | tuple[float, float], _crs: int,
                      _save_dir: Path, _all_touched: bool) -> DataArray:
    """
    Process the base rule of the rules DataFrame for the resolution `_res` and the `_crs`
    in a `_all_touched` fashion or not.
    and save the raster
    at `_save_dir`
    :param df: DataFrame with all rules
    :param _res: float or tuple of two floats of the resolution
    :param _crs: the CRS as EPSG-Code
    :param _save_dir: save path of the raster
    :param _all_touched: bool
    :return: raster as DataArray
    """
    nmd_tpl = df.copy().loc[df['Description'] == 'Base'].iloc[0]
    vec = __vector_processing(nmd_tpl, _crs)
    save_path = _save_dir / f"res_{_res}/all_touched_{_all_touched}/rule_{nmd_tpl.LineNum}_{nmd_tpl.Description}.tif"
    raster = make_geocube(vec, ['Weights', ], resolution=_res, fill=999,
                          rasterize_function=partial(rasterize_image, all_touched=_all_touched))
    raster = raster['Weights']

    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_compressed(raster, save_path)
    return raster


def process_rule(nmd_tpl, example_da: DataArray, _crs: int,
                 _save_dir: Path, _all_touched: bool, bbox) -> list[str | Path]:
    """
    Process the one single line of the rules DataFrame and return the save path in a list
    (so that ic can be concat easily)
    :param nmd_tpl: namedTuple
    :param example_da: example raster as DataArray
    :param _crs: crs as EPSG-Code
    :param _save_dir: save dir of the raster
    :param _all_touched: Bool process with all_touched
    :param bbox: bounding box, a tuple of 4 floats
    :return: list[str| Path]
    """
    # resolution from example_Data
    res = float(abs(example_da.rio.transform()[0]))

    save_path = _save_dir / f"res_{res}/all_touched_{_all_touched}/rule_{nmd_tpl.LineNum}_{nmd_tpl.Description}.tif"
    if not save_path.exists():  # reuse existing results
        vec = __vector_processing(nmd_tpl, _crs, bbox)
        raster = make_geocube(vec, ['Weights'], like=example_da, fill=999,
                              rasterize_function=partial(rasterize_image, all_touched=_all_touched))
        raster = raster['Weights']
        raster.data[example_da == example_da.rio.nodata] = raster.rio.nodata
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_compressed(raster, save_path)
    return [save_path, ]


if __name__ == '__main__':
    parser = ArgumentParser(description='Read in Vectors from Configuration file and rasterize them.')
    parser.add_argument('config_file_path', type=str, help='Full path to the .xlsx config file.')
    parser.add_argument('vectors_main_folder', type=str, help='Main Path to the vector files.')
    parser.add_argument('save_dir', type=str, help='Path where this projects raster files can the saved')
    parser.add_argument('-r', '--resolution', type=float, default=1000, help='Resolution to use.')
    parser.add_argument('-at', '--all_touched', action='store_true', help='Select those pixels, those CENTER  is '
                                                                          'covered by the Polygon (False) or any '
                                                                          'part overlaps with the Polygon (True).')
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

    # combine the raster's file
    # finalized_rules -> max, finalized_rules + default (where processed is nodata)
    finalized_rules = merge_arrays([open_rasterio(x, cache=False) for x in finalized_rules], method='max')

    finalized_rules.data = np.where(finalized_rules == finalized_rules.rio.nodata, base_raster, finalized_rules)
    write_compressed(finalized_rules, save_dir / f'result_res_{resolution}_all_touched_{all_touched}.tif')
    write_png(finalized_rules, save_dir / f'result_res_{resolution}_all_touched_{all_touched}.png')
