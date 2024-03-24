# Use a base image with the necessary dependencies
FROM public.ecr.aws/lambda/python:3.11


# Copy the requirements file and install dependencies
RUN pip install pandas
RUN pip install boto3
RUN pip install tensorflow
RUN pip install scikit-learn

# Copy the rest of the application code into the container
COPY predict.py ./

ENV BUCKET_NAME="data-608-project-private"

# Specify the command to run your training script
CMD ["predict.lambda_handler"]
