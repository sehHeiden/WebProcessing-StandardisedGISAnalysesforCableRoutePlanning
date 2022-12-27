from pywps import Service
from pathlib import Path

from src.wps.processes.least_cost_path import LeastCostPath
config = r'./src/wps/wsgi/pywps_wsgi.cfg'
print(Path(config).exists())

application = Service([LeastCostPath(), ], [config, ])
