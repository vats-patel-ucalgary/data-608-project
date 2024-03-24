# Churn Prediction

---

#### Technologies Used

- AWS Lambda
- AWS ECR (Elastic Container Registry)
- AWS S3 for storage
- AWS Streamlit
- Docker for hosting and container building
- Python as coding language
- Tensorflow/keras for training model

#### Setup

**Install docker in your system**

> Note: Before building change the environment variables according to your requirements

```sh
# Training Model
docker build . -t data-608-project:train -f Training/Dockerfile
docker tag data-608-project:train username/data-608-project:train
docker push username/data-608-project:train

# Predicting
docker build . -t data-608-project:predict -f Prediction/Dockerfile
docker tag data-608-project:predict username/data-608-project:predict
docker push username/data-608-project:predict

# Streamlit
docker build . -t data-608-project:streamlit -f Streamlit/Dockerfile
docker tag data-608-project:streamlit username/data-608-project:streamlit
docker push username/data-608-project:streamlit
```

> Note: **Create ECR Repository** with **data-608-project** name

**Login to EC2**

> **Note:** Here We are using ec2 as we are working in LabRole and we don't have any API credentials for s3 directly, so we are pushing and pulling the docker image to EC2 instance. You can directly push to ECR from you local system if you have the credentials

```sh
# Training Model
docker pull username/data-608-project:train
docker tag username/data-608-project:train ecr.xxxx.com/data-608-project:train
docker push ecr.xxxx.com/data-608-project:train

# Prediction
docker pull username/data-608-project:predict
docker tag username/data-608-project:predict ecr.xxxx.com/data-608-project:predict
docker push ecr.xxxx.com/data-608-project:predict
```

**Create Lambda function**

- Using container option and select images with tagname (train) for trainModel (lambda) and tagname (predict) for predictModel (lambda).
- For predict model give API Gateway as a trigger which is mentioned in the Dockerfile of Streamlit.
- Run trainingModel (lambda) funciton manually to train the model which create two pickle files and stores them in S3

**Login to EC2 again**

> **Note:** Here as an alternative of ec2 you can use any of the other platform to host the streamlit app. But as we have LabRole assign for AWS, we have to use Ec2 for access of S3 bucket.

**_Run Streamlit using docker_**

```sh
# Streamlit
docker pull username/data-608-project:streamlit
docker run -p 8501:8501 -d username/data-608-project:streamlit
```

**See the url on which streamlit is running**

```sh
docker logs id_of_docker_containg
```

Copy/paste the url in new browser window
