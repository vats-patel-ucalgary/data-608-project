import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import boto3
from io import StringIO

bucket_name = 'data-608-project-private'
file_key = 'Churn_Modelling.csv'

# Create an S3 client
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')
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

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Splitting data manually

X_test = X[:4]  # first 4 rows for test set
X_train = X[4:]  # rest for training set
y_test = y[:4]  # corresponding labels for test set
y_train = y[4:]  # corresponding labels for training set

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Path to save the model object
ann_path = 'model.bin'

# Path to save the label encoder object
le_path = 'le.bin'

# Path to save the Column transformer object
ct_path = 'ct.bin'

# Path to save the Standard Scalar object
sc_path = 'sc.bin'

# Writing the model to the pickle file
ann_pickle_obj = pickle.dumps(ann)
s3_resource.Object(bucket_name, ann_path).put(Body=ann_pickle_obj)
print('ann written success.')

# Writing the label encoder to the pickle file
le_pickle_obj = pickle.dumps(le)
s3_resource.Object(bucket_name, le_path).put(Body=le_pickle_obj)
print('le written success.')

# Writing the column transformer object to the pickle file
ct_pickle_obj = pickle.dumps(ct)
s3_resource.Object(bucket_name, ct_path).put(Body=ct_pickle_obj)
print('ct written success.')

# Writing the standard scalar to the pickle file
sc_pickle_obj = pickle.dumps(sc)
s3_resource.Object(bucket_name, sc_path).put(Body=sc_pickle_obj)
print('sc written success.\n')

print(f"X_test: {X_test}")
