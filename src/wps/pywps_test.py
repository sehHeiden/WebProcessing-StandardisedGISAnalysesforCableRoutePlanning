import flask
import pywps
from os.path import join, splitext, isfile
from .processes.least_cost_path import LeastCostPath


service = pywps.Service([LeastCostPath()], ['./src/wps/pywps.cfg', ])
app = flask.Flask(__name__)
app.route('/wps', methods=['GET', 'POST'])(lambda: service)


@app.route('/outputs/'+'<path:filename>')
def outputfile(filename):
    target_file = join('outputs', filename)
    if isfile(target_file):
        file_ext = splitext(target_file)[1]
        with open(target_file, mode='rb') as f:
            file_bytes = f.read()
        mime_type = None
        if 'xml' in file_ext:
            mime_type = 'text/xml'
        return flask.Response(file_bytes, content_type=mime_type)
    else:
        flask.abort(404)


if __name__ == '__main__':
    app.run()
