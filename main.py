from flask import Flask, request, render_template, url_for
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np
import random
import os

app = Flask(__name__)
CUR_DIR = os.getcwd()

# import the data
dataset = pd.read_csv(CUR_DIR + '/athlete_classification.csv')
# set up feature columns
feature_columns = ['Sex', 'Age', 'Height', 'Weight']
X = dataset[feature_columns].values
Y = dataset['Sport'].values

# code to visualize data during tweaking of hyper-parameters
"""
plt.figure(figsize=(15,10))
parallel_coordinates(dataset.drop("Id", axis=1), "Medal")
plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Features values', fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
"""

# Splitting into training and test datasets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.7, random_state = 0)
# Creating the learning model
knn_classifier = KNeighborsClassifier(n_neighbors=6)
# Fitting the model with the training data
knn_classifier.fit(X_train, Y_train)
# Making predictions with the test data (This line is also where we would potentially classify new data)
Y_pred = knn_classifier.predict(X_test)
# Finding Accuracy:
accuracy = accuracy_score(Y_test, Y_pred)*100
print('Accuracy of model: ' + str(round(accuracy, 2)) + ' %.')

# code to visualize data during tweaking of hyper-parameters
"""
# creating list of cv scores
cv_scores = []
# perform 10-fold cross validation
for k in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Displaying results visually
plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.plot(list(range(1,40)), cv_scores)
plt.show()
"""

@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/handle_input", methods=["GET"])
def handle_input():
    classes = ["Gymnastics", "Swimming", "Athletics"]
    try:
        values = [[float(request.args.get(component)) for component in ['gender', 'age', 'height', 'weight']]]
    except TypeError:
        return "ERROR: Need an age, height, weight, and gender parameter"

    prediction = knn_classifier.predict(values)[0]
    return classes[prediction]

app.run(debug=True)
