import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the data from the Excel spreadsheet
df = pd.read_excel('data.xlsx')

# Clean the data by removing any rows where the event type is 'unknown'
df = df[df['event_type'] != 'unknown']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :-1], df.iloc[:, -1], test_size=0.3, random_state=42)

# Train a logistic regression classifier with a high decision threshold for the 'event' class
clf = LogisticRegression(class_weight={'event': 10, 'non-event': 1}, C=0.1)
clf.fit(X_train, y_train)

# Make predictions on the test set and print the classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))