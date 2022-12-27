import flask
import pywps
from .processes.least_cost_path import LeastCostPath
from pathlib import Path


service = pywps.Service([LeastCostPath()], ['./src/wps/pywps_flask.cfg', ])
app = flask.Flask(__name__)
app.route('/wps', methods=['GET', 'POST'])(lambda: service)


@app.route('/tmp/'+'<path:filename>')
def outputfile(filename: str):
    """
    Send the files with the given json file in folder `/tmp`.
    `/tmp` has to be configured in `.cfg` file for PyWPS as outputpath.
    :param filename: str, name of the json file
    :return: falsk.Resonse or abort (None) transmitting the input json file
    """
    print(filename)
    target_file = Path('/tmp') / filename
    if target_file.is_file():
        file_ext = target_file.suffix
        with open(target_file, mode='rb') as f:
            file_bytes = f.read()
        mime_type = None
        if 'json' in file_ext:
            mime_type = 'text/json'
        return flask.Response(file_bytes, content_type=mime_type)
    else:
        flask.abort(404)


if __name__ == '__main__':
    app.run()
