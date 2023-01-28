# Documentation of PyWPS

## Installation
For install use pip or conda. No need to git clone and self build. Could not build PyWPS myself. 
This way installing the dependencies is not needed. Some dependencies do not exist on modern
Ubuntu (renamed?)

## Server
When using flask the standard test site runs on localhost using port 5000: `http://127.0.0.1:5000`.
Although flask is not in the dependencies, it has to be installed via pip or conda.
Pro for flask is: it worked ony LINUX from receiving data to sending the correct reply. 
Not on Windows. On Windows the temporary folder for pywps receiving the data is empty.
Alternative to flask is using WSGI.
PyWPS documentation describes Apache and gunicorn.  
Apache is not working for me. 
The Gunicorn package is not named gunicorn3 anymore (after Ubuntu 16). 
In contrast to the documentation it's better to install gunicorn from pip/conda, so that it can be used inside the conda environment.
Perhaps, it's different in docker and using a conda environment.
Gunicorn can easily bind to a Layer 4 address. I used  `http://127.0.0.1:8081`. 
But I did get it sending a reply running.

For Flask, routes have to be creates for downloading and running the server. The used routes are /tmp/{filename.json}
and /wps. For configuration of the server PyWPS uses .cfg-Files for flask and WSGI. 
When using the `/tmp`-directory to save the output files and send these:

```
outputpath = /tmp
outputurl = http://<ip-adress>:<port>/tmp
```

has to be set. This way I was able to configure the WPS with flask. But not with wsgi.

## Processes
For every capability of the WPS a subclass of the PyWPS `Process` class has to be created.
The `__init__` method defines the input and output data. The `_handle` method is used to
process the data.


## Capabilities
+ get capabilities: http://127.0.0.1:5000/wps?service=WPS&request=GetCapabilities&version=1.0.0
+ describe process: http://127.0.0.1:5000/wps?service=WPS&request=DescribeProcess&version=1.0.0&identifier=<CAPABILITY>
+ execute process: [Used birdy]

## Sending receiving data

The sending and receiving data has successfully been tested with the python library [birdy](https://birdy.readthedocs.io/en/latest/).
which builds on-top of the library owslab.wps. Sending/Receiving the data with owslab.wps did not work. To complicated to
send own data?
I also tested the QGIS Plug-in [WPSCLient](https://plugins.qgis.org/plugins/wps/) The last stable release was 2017
only works with QGIS 2. The last QGIS2 release was 2019. I tried The plug-in but was not able to send my own files to
the server.
