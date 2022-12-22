from pywps import Service

from src.wps.processes.least_cost_path import LeastCostPath

application = Service([LeastCostPath()], ['./pywps.cfg', ])
