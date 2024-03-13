# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

obj = 5

# Writing the standard scalar to the pickle file

# with open(sc_path, 'wb') as f:
#     pickle.dump(sc, f)
#     print('sc written success.')
#     f.close()
import boto3
bucket_name = 'data-608-project-private'

dataset_pickle_obj = pickle.dumps(obj)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket_name, 'test2.bin').put(Body=dataset_pickle_obj)

print(f"SUCCESS UPLOAD")
# dataset = pd.read_csv('Churn_Modelling.csv')
# print(dataset.head(5))

# X = dataset.iloc[:, 3:-1].values
# y = dataset.iloc[:, -1].values
# print(X)
# print(y)

# # for gender column
# le = LabelEncoder()
# X[:, 2] = le.fit_transform(X[:, 2])

# # for the geography column
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# print(X)

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# # Splitting data manually

# X_test = X[:4]  # first 4 rows for test set
# X_train = X[4:]  # rest for training set
# y_test = y[:4]  # corresponding labels for test set
# y_train = y[4:]  # corresponding labels for training set

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# ann = tf.keras.models.Sequential()
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# # Path to save the model object
# ann_path = 'model.bin'

# # Path to save the label encoder object
# le_path = 'le.bin'

# # Path to save the Column transformer object
# ct_path = 'ct.bin'

# # Path to save the Standard Scalar object
# sc_path = 'sc.bin'

# # Writing the model to the pickle file
# with open(ann_path, 'wb') as f:
#     pickle.dump(ann, f)
#     print('ann written success.')
#     f.close()

# # Writing the label encoder to the pickle file
# with open(le_path, 'wb') as f:
#     pickle.dump(le, f)
#     print('le written success.')
#     f.close()

# # Writing the column transformer object to the pickle file
# with open(ct_path, 'wb') as f:
#     pickle.dump(ct, f)
#     print('ct written success.')
#     f.close()

# # Writing the standard scalar to the pickle file
# with open(sc_path, 'wb') as f:
#     pickle.dump(sc, f)
#     print('sc written success.')
#     f.close()

# # OR

# # Store to S3

# print(f"X_test: {X_test}")