from birdy import WPSClient
from pathlib import Path
from json import load
from argparse import ArgumentParser
from metalink import download


if __name__ == '__main__':
    parser = ArgumentParser(description='Find the nearest cost path with a wps using birdy.')
    parser.add_argument('config_path', type=str, help='Full path to the config file. Containing relative paths.')
    parser.add_argument('main_path', type=str, help='Main path for the files in the config.')
    args = parser.parse_args()

    main_path = Path(args.main_path)
    with open(args.config_path, 'r') as f:
        config = load(f)

    pywps = WPSClient('http://localhost:5000/wps')

    cost_raster = main_path / config['cost_raster']
    start_features = main_path / config['start_features']
    end_features = main_path / config['end_features']

    result = pywps.lcp(costs=cost_raster,
                       start=start_features,
                       end=end_features)
    print(result.get(asobj=True)[0])  # print the geojson?

    f_name = str(result.get(asobj=False)[0])
    print(f_name)
    download.get(f_name, path='.', segmented=False)
