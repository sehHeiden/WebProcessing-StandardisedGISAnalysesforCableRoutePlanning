from pywps import Service
from pathlib import Path

from src.wps.processes.least_cost_path import LeastCostPath
config = r'./src/wps/pywps.cfg'
print(Path(config).exists())

application = Service([LeastCostPath(), ], [config, ])
