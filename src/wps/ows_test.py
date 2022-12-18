from owslib.wps import WebProcessingService, GMLMultiPolygonFeatureCollection, printInputOutput, ComplexDataInput

client = WebProcessingService('http://127.0.0.1:5000/wps')
client.getcapabilities()
print(client.processes)

process_id = 'MyBuffer'

buffer_process_description = client.describeprocess(process_id)
for i in buffer_process_description.dataInputs:
    printInputOutput(i)

# polygon = [(-102.8184, 39.5273), (-102.8184, 37.418), (-101.2363, 37.418), (-101.2363, 39.5273), (-102.8184, 39.5273)]
in_data = ComplexDataInput(r"E:\Basti\Studium\HSHarz\WissenschaftlichesArbeiten\WebProcessing-StandardisedGISAnalysesforCableRoutePlanning\results\least_cost_paths\least_cost_path_test_points_res_10_al_false.gml")
# featureCollection = GMLMultiPolygonFeatureCollection([polygon, ])

inputs = [("poly_in", in_data),
          # ("distance", 10.0),
          ]

execution = client.execute(process_id, inputs, output="OUTPUT")
