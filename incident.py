import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Read the data
df = pd.read_excel('incident_data.xls')

# Select only the relevant columns
df = df[['incident_state', 'category']]

# Fill the missing values
df = df.fillna('')

# Filter out the rows with missing short description
df = df[df['incident_state'] != '']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['incident_state'], df['category'], random_state=42)

# Convert the text into a matrix of token counts
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

# Train the model
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Evaluate the model
X_test_counts = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classify new incidents
new_incidents = [
    'New',
    'Resolved',
    'Active',
    'Closed'
]
X_new_counts = vectorizer.transform(new_incidents)
y_new_pred = clf.predict(X_new_counts)
print(f'New incident categories: {y_new_pred}')

# Add predicted subcategories to product name column
df['Productname'] = df.apply(lambda x: x['category'] if x['category'] in x['Productname'] else x['Productname'] + ' - ' + x['category'], axis=1)

# Save the results to a new Excel file
df.to_excel('classified_incidents.xlsx', index=False)
