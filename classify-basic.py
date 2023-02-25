#
# BASIC CLASSIFICATION ML MODEL
# By Adrian Mikula
#
# This model takes a simple dataset and classifies it into two categories.
# It will not work well for more than two categories.
#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Specify the dataset to load
dataset_filename = 'Viv all waveforms & parameters'

# Specify the target variable or label column
target_column = 'Imported Tag'

# Load the data from the Excel spreadsheet
print("Loading the dataset")
df = pd.read_excel('datasets/' + dataset_filename + '.xlsx', nrows=10000,
                   usecols=lambda x: x != 'Date - Time' and x != 'Imported Tag Meaning')

# Drop rows with any missing values
print("Cleaning the dataset")
df = df.dropna()

# TODO Clean the data by removing any rows where the event type is 'unknown'
#df = df[df['event_type'] != 'unknown']

# Create a set of unique categories
categories = set(df[target_column])

# Map the categorical column to an integer column
df[target_column] = df[target_column].apply(lambda x: list(categories).index(x))

# Split the data into training and testing sets
print("Splitting the data")
x = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

# train the model
print("Training the model")

# Define the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict the classes of the test data
print("Testing the model")
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
print("")
print("Confusion Matrix")
print("TN, FP")
print("FN, TP")
print(confusion_matrix(y_test, y_pred))
print("")
print(classification_report(y_test, y_pred))