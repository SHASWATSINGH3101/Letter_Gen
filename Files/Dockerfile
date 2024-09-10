# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create directories for the cache and flagged data, and set permissions
RUN mkdir -p /app/.cache /app/flagged && chmod 777 /app/.cache /app/flagged

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Set environment variable for the Hugging Face home directory
ENV HF_HOME=/app/.cache

# Run app.py when the container launches
CMD ["python", "app.py"]
