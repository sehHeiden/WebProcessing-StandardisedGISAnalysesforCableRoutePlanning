from pywps import Process, ComplexInput, ComplexOutput, Format
from rioxarray import open_rasterio
from geopandas import read_file
from json import dumps

from src.least_cost_path.find_least_cost_path import find_least_cost_path


class LeastCostPath(Process):
    def __init__(self):
        inputs = [ComplexInput('costs', 'Cost Raster', supported_formats=[Format('image/tiff'), ]),
                  ComplexInput('start', 'Starting Point',
                               supported_formats=[Format('application/gpkg'), Format('application/json'), ]),
                  ComplexInput('end', 'Ending Point',
                               supported_formats=[Format('application/gpkg'), Format('application/json'), ])]
        outputs = [ComplexOutput('out', 'Referenced Output',
                                 supported_formats=[
                                     Format('application/json')
                                 ])]

        super(LeastCostPath, self).__init__(
            self._handler,
            identifier='lcp',
            title='Process least cost path',
            abstract='Returns a GeoJSON \
                with with least cost path from cost raster.',
            inputs=inputs,
            outputs=outputs,
            store_supported=True,
            status_supported=True
        )

    def _handler(self, request, response):
        input_cost_raster = open_rasterio(request.inputs['costs'][0].file)
        input_start = read_file(request.inputs['start'][0].file)
        input_end_points = read_file(request.inputs['end'][0].file)

        lcp = find_least_cost_path(input_cost_raster, 0, False, input_start, input_end_points)

        response.outputs['out'].data = lcp.to_json(indent=2)
        return response
