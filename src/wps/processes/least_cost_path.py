from pywps import Process, ComplexInput, ComplexOutput, Format, FORMATS
from rioxarray import open_rasterio
from geopandas import read_file

from src.least_cost_path.find_least_cost_path import find_least_cost_path


class LeastCostPath(Process):
    """
    Process Class to compute the least cost path from input.
    __init__ describes the data.
    _handle computes the least cost path.
    Class instiacted by server.
    """
    def __init__(self):
        inputs = [ComplexInput('costs', 'Cost Raster', supported_formats=[Format('image/tiff'), ]),
                  ComplexInput('start', 'Starting Point',
                               supported_formats=[Format('application/gpkg'), Format('application/json'), ]),
                  ComplexInput('end', 'Ending Point',
                               supported_formats=[Format('application/gpkg'), Format('application/json'), ])]
        outputs = [ComplexOutput('out', 'Referenced Output',
                                 as_reference=True,
                                 supported_formats=[FORMATS.JSON, ]
                                 )]

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

        # start
        response.update_status('Least Cost Path Process started.', 0)
        lcp = find_least_cost_path(input_cost_raster, 0, False, input_start, input_end_points)

        response.outputs['out'].data = lcp.to_json(indent=2)
        response.update_status('Least Cost Path Process completed.', 100)
        return response
