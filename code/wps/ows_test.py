from owslib.wps import WebProcessingService, GMLMultiPolygonFeatureCollection, printInputOutput

client = WebProcessingService('http://127.0.0.1:5000/wps')
client.getcapabilities()
print(client.processes)

process_id = 'buffer'

buffer_process_description = client.describeprocess(process_id)
for i in buffer_process_description.dataInputs:
    printInputOutput(i)

polygon = [(-102.8184, 39.5273), (-102.8184, 37.418), (-101.2363, 37.418), (-101.2363, 39.5273), (-102.8184, 39.5273)]
featureCollection = GMLMultiPolygonFeatureCollection([polygon])

inputs = [("poly_in", featureCollection),
          ("buffer", 10.0), ]

execution = client.execute(process_id, inputs, output="OUTPUT")
