import flask
import pywps

from .processes.least_cost_path import LeastCostPath

service = pywps.Service([LeastCostPath()], ['./src/wps/pywps.cfg', ])
app = flask.Flask(__name__)
app.route('/wps', methods=['GET', 'POST'])(lambda: service)

if __name__ == '__main__':
    app.run()
