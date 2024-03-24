# Use a base image with the necessary dependencies
FROM python:3.11.0

WORKDIR /app

# Copy the requirements file and install dependencies
RUN pip install streamlit
RUN pip install pandas
RUN pip install boto3

# Copy the rest of the application code into the container
COPY . .

ENV API_ENDPOINT="https://4yy4hvhc93.execute-api.us-east-1.amazonaws.com/default/predictModel"
ENV BUCKET_NAME="data-608-project-private"

# Specify the command to run your training script
CMD ["streamlit", "run", "streamlit.py"]
