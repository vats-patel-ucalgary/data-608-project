import pickle
import boto3
import pandas as pd
import keras
import json
import os

def lambda_handler(event, context):

  train_data = None

  s3_client = boto3.client('s3')
  bucket_name = os.getenv('BUCKET_NAME')

  response = s3_client.get_object(Bucket=bucket_name, Key='train.bin')

  # Read the contents of the file
  model_file_content = response['Body'].read()
  train_data = pickle.loads(model_file_content)

  json_model = train_data['ann']
  le = train_data['le']
  ct =  train_data['ct']
  sc =  train_data['sc']

  ann = keras.models.model_from_json(json_model)
  s3_client.download_file(bucket_name, 'model.weights.h5', '/tmp/model.weights.h5')

  # load weights into new model
  ann.load_weights("/tmp/model.weights.h5")
  print("Loaded model from disk")

  # Read the contents of the file
  input_data =  json.loads(event['body'])

  dataset=pd.json_normalize(input_data)

  X = dataset.values

  user_input = X[0]

  def tranform_user_input(X_input):
    X_input = X_input.copy()

    X_input[2] = le.transform([X_input[2]])[0]
    X_input = ct.transform(X_input.reshape(1, -1))
    X_input = sc.transform(X_input)
    return X_input

  user_input = tranform_user_input(user_input)
  print(user_input)
  y_pred = ann.predict(user_input)
  print('Predict success')

  return {
    'statusCode': 200,
    'body': json.dumps({
      'data': str(y_pred[0][0])
    })
  }