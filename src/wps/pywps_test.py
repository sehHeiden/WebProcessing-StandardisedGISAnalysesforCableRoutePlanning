import flask
import pywps

from .processes.buffer import Buffer, MyBuffer
from .processes.centroids import Centroids
from .processes.least_cost_path import LeastCostPath

service = pywps.Service([Buffer(), MyBuffer(), Centroids(), LeastCostPath()], ['pywps.cfg', ])
app = flask.Flask(__name__)
app.route('/wps', methods=['GET', 'POST'])(lambda: service)

if __name__ == '__main__':
    app.run()
