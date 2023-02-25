import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# Load the data from the Excel spreadsheet
print("Loading the dataset")
df = pd.read_excel('datasets/Viv all waveforms & parameters.xlsx', nrows=10000,
                   usecols=lambda x: x != 'Date - Time' and x != 'Imported Tag Meaning')

# Drop rows with any missing values
print("Cleaning the dataset")
df = df.dropna()

# TODO Clean the data by removing any rows where the event type is 'unknown'
#df = df[df['event_type'] != 'unknown']

# Convert the categorical column into multiple binary columns
dummies = pd.get_dummies(df['Imported Tag'], prefix='category', dtype=int)

# Concatenate the binary columns with the original dataframe
df = pd.concat([df, dummies], axis=1)
df.drop('Imported Tag', axis='columns', inplace=True)

# Split the data into training and testing sets
print("Training the models")
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :-1], df.iloc[:, -1], test_size=0.3, random_state=42)

# Define the classifiers to compare
classifiers = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Support Vector Machine', SVC(kernel='linear', n_jobs=4)),
    ('Logistic Regression', LogisticRegression())]

# Evaluate the performance of each classifier using cross-validation
for clf_name, clf in classifiers:
    print("Training the " + clf_name + " model")
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f'{clf_name} - Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})')
