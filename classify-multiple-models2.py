import numpy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

# Convert the categorical column into multiple binary columns
dummies = pd.get_dummies(df['Imported Tag'], prefix='category', dtype=int)

# Concatenate the binary columns with the original dataframe
# df = pd.concat([df, dummies], axis=1)
df.drop('Imported Tag', axis='columns', inplace=True)

# Split the data into training and testing sets
print("Splitting the data")

# x = df.drop('target_column', axis=1)
x = df
# y = df[['category', 'category', 'category']]
# y = dummies
# y = dummies[['category_b', 'category_e']]
y = dummies.filter(regex='^category_', axis=1)

# Convert the target variable to a one-dimensional array
# y = numpy.squeeze(y)
# y = y.values.reshape(-1)
# Convert the binary columns to a one-dimensional array
print("input data preview: " + str(x.iloc[0]))
print("output data preview: " + str(y.iloc[0]))
y = y.to_numpy().argmax(axis=1)

# print("output data preview: " + str(y.iloc[0]))
print("output data preview: " + str(y))

# train the model
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)


# Define the classifiers to compare
print("Training the models")

classifiers = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    # ('Support Vector Machine', SVC(kernel='linear')),
    ('Decision Tree Classifier',  DecisionTreeClassifier()),
    ('Logistic Regression', LogisticRegression())]

# Evaluate the performance of each classifier using cross-validation
for clf_name, clf in classifiers:
    print("")
    print("Training the " + clf_name + " model")
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f'{clf_name} - Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})')
