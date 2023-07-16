# Use an official Python runtime as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /training_pipeline

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the training pipeline code into the container
COPY ./model.py /training_pipeline
COPY ./utils.py /training_pipeline

# Run the training script
CMD ["python", "-m", "model"]
