import flask
import pywps

from .buffer import Buffer
from .centroids import Centroids


service = pywps.Service([Buffer(), Centroids()])
app = flask.Flask(__name__)
app.route('/wps', methods=['GET', 'POST'])(lambda: service)
app.run()
