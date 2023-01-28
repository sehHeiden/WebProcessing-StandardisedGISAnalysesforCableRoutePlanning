from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from json import load
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fiona.errors import UnsupportedGeometryTypeError
from geopandas import read_file, GeoDataFrame
from pandas import concat, date_range
from requests import get, codes
from requests.exceptions import ReadTimeout, ConnectionError
from shapely.geometry import Point
from tqdm import tqdm
from warnings import warn


def download_zips(_url: str, _save_path: Path) -> bool:
    try:
        response = get(_url, timeout=(2, 5))
    except (ConnectionError, ReadTimeout):
        return False
    if isinstance(response.status_code, int) and response.status_code == codes.ok:
        with ZipFile(BytesIO(response.content)) as source_zip:
            source_zip.extractall(_save_path)
        return True
    else:
        return False


def concat_vector_files(in_dir: Path) -> GeoDataFrame:
    lod1_vectors = []
    shp_files = [str(x) for x in in_dir.iterdir() if x.suffix == '.shp']
    error_counter = 0
    for f in shp_files:
        try:
            lod1_vectors.append(read_file(f))
        except UnsupportedGeometryTypeError:
            error_counter = error_counter + 1
    if error_counter > 0:
        warn(f'{error_counter} could not be concatenated.')

    return concat(lod1_vectors)


if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('json', type=str, help='Full path to the json config file.')

    args = parser.parse_args()

    with open(args.json) as j:
        config = load(j)

    geb_path = config['areas_vector_path']

    geb_vector = read_file(geb_path)
    geb_vector = geb_vector.loc[geb_vector.BEZ_LAN == 'Niedersachsen']
    geb_vector = geb_vector.loc[~geb_vector.BEZ_KRS.isna()]
    geb_vector = geb_vector.dissolve(by='BEZ_LAN')

    x_range = list(range(2342, 2672, 2))
    y_range = range(5684, 5970, 2)

    unzip_path = Path(config['unzip_lod1_path'])
    dates = date_range(*config['dates'], freq='1D', inclusive='both')

    loaded_files_count = np.count_nonzero([True for x in unzip_path.iterdir() if x.suffix == '.shp'])
    possible_files = np.count_nonzero([True for y in y_range for x in x_range if
                                       np.any(geb_vector.intersects(Point(x * 1_000 - 2_000_000, y * 1_000)))])

    download_page = config['download_page']

    p_download = partial(download_zips, _save_path=unzip_path)
    # down load LoD1
    for dt in tqdm(dates, position=0, leave=True, desc='Days'):
        d = dt.date()

        p_bar = tqdm(x_range, position=1, leave=False)
        for x in p_bar:
            
            lf = loaded_files_count
            pf = possible_files
            p_bar.set_description(f"{lf} of {pf} ({(lf / pf * 100):3.1f} %) new files downloaded.")

            # only when point valid and in Lower Saxony
            ys_intersecting = [y for y in y_range if
                               np.any(geb_vector.intersects(Point(x * 1_000 - 2_000_000, y * 1_000)))]
            ys_intersecting = [y for y in ys_intersecting if
                               not np.any([s.stem.startswith(f"LoD1_{x}{y}") for s in unzip_path.iterdir()])]
            urls = [f"{download_page}/{x}{y}/{d}/LoD1_{x}{y}_2_{d}.zip" for y in ys_intersecting]

            with ThreadPoolExecutor() as p:
                is_downloaded = np.sum(np.fromiter(p.map(p_download, urls),
                                                   bool), )
                loaded_files_count = loaded_files_count + is_downloaded

    concat_vector_files(unzip_path).to_file(config['save_result_path'])
