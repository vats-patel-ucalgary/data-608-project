# Use a base image with the necessary dependencies
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run your training script
CMD ["python", "train.py"]
