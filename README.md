# HCSBootcamp2
The coolest Olympic sport predictor of all time.
# Usage
Run "main.py" to start the flask server. From there, navigate to "http://127.0.0.1:5000/predict_sport?age=NUM&height=NUM&weight=NUM&sex=NUM" to submit requests directly or navigate to "http://127.0.0.1:5000/" to use a basic interface to make requests.
# Data
https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results
I took 500 random rows for athletes in Swimming, Gymnastics, and Athletics.
# Motivation
Originally, I was hoping to use age, height, weight, and sex to predict whether an athlete would place on the podium. However, the functionality was poor because it predicted non-podium almost %100 of the time, which isn't interesting. So I decided to pivot and instead try and classify what sport an athlete played based on their physical characteristics. It's interesting because it allows you to make quick and easy observations about the different builds of Olympic athletes, and what event you might be best suited for yourself.
