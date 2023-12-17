# Dockerfile for Time series Analysis App deployment
# ----------------------------------------------------------
# This Dockerfile is used to build a Docker image for running
# an Ensemble learning of Time Series Forecastin app. 
# It starts with an Alpine-based Python 3.10 image, 
# sets up the working directory,
# installs the Python dependencies listed in requirements.txt,
# copies the application source code into the container,
# exposes port 8501 to allow external access, and finally,
# runs the app using the specified command.
# ----------------------------------------------------------

FROM python:3.10.0-alpine

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["python", "src/main.py", "src/Features/all_data_ts.csv"]