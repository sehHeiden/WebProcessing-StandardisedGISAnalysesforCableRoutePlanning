FROM python:3.10-slim
WORKDIR /app
# By copying over requirements first, we make sure that Docker will cache
# our installed requirements rather than reinstall them on every build
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
# Now copy in our code, and run it

# CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root