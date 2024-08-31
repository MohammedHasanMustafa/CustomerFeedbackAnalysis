# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mlflow

# Copy the rest of the application code
COPY . .

# Specify the command to run on container start
CMD ["python", "app.py"]
