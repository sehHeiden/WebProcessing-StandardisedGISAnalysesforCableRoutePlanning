from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from json import load
from pathlib import Path

from geopandas import read_file
from requests import get

from download_lod1 import download_zips, concat_vector_files

if __name__ == '__main__':
    parser = ArgumentParser(description='Download all files ')
    parser.add_argument('json', type=str, help='Full path to the json config file.')

    args = parser.parse_args()

    with open(args.json) as j:
        config = load(j)

        geb_path = config['areas_vector_path']
        response = get(config['mass_geojson_url']).content
    tmp = BytesIO(response)
    all_files_gdf = read_file(tmp)
    unzip_path = Path(config['unzip_lod1_path'])

    p_download = partial(download_zips, _save_path=unzip_path)

    existing_files = [x for x in unzip_path.iterdir() if x.suffix == ".shp"]
    existing_tiles_names = [x.stem.split('_')[1] for x in existing_files]

    to_download_gdf = all_files_gdf.loc[~all_files_gdf['tile_id'].isin(existing_tiles_names)]
    with ThreadPoolExecutor() as p:
        is_downloaded = p.map(p_download, to_download_gdf['3DShape'])

    concat_vector_files(unzip_path).to_file(config['save_result_path'])

