import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import boto3
from io import StringIO
import os

def lambda_handler(event, context):
    bucket_name = os.getenv('BUCKET_NAME')
    file_key = 'Churn_Modelling.csv'

    # Create an S3 client
    s3_client = boto3.client('s3')
    s3_resource = boto3.resource('s3')
    file_content = None
    try:
        # Retrieve the file object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)

        # Read the contents of the file
        file_content = response['Body'].read().decode('utf-8')
    except Exception as e:
        print(f'Error in reading csv file.')

    dataset = pd.read_csv(StringIO(file_content))
    print(dataset.head(3))

    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values
    print(X)
    print(y)

    # for gender column
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])

    # for the geography column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    ann = keras.models.Sequential()
    ann.add(keras.layers.Dense(units=6, activation='relu'))
    ann.add(keras.layers.Dense(units=6, activation='relu'))
    ann.add(keras.layers.Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

    train_data = {
        'le': le,
        'ct': ct,
        'ann': ann.to_json(),
        'sc': sc
    }

    # serialize weights to HDF5
    ann.save_weights("/tmp/model.weights.h5")
    s3_resource.Bucket(bucket_name).upload_file("/tmp/model.weights.h5", "model.weights.h5")

    train_data_pickle_obj = pickle.dumps(train_data)
    s3_resource.Object(bucket_name, 'train.bin').put(Body=train_data_pickle_obj)
    print('Train data written success.\n')

    return {
        'success': 'Trained successfully.'
    }