# Use a base image with the necessary dependencies
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
RUN pip install pandas
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install boto3
RUN pip install tensorflow

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run your training script
CMD ["python", "train.py"]
