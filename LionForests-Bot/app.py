import sys
sys.path.append(sys.path[0]+'\\LionForests')
from flask import Flask, render_template, request, jsonify
from LionForests import LionForests
from utilities.dummy_utilizer import DummyUtilizer
from LFBot import LFBot
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import urllib
import warnings


warnings.filterwarnings("ignore")

student = pd.read_csv('students2.csv')
feature_names = list(student.columns)[:-1]
class_names=["Won't graduate",'Will graduate (eventually)']
X = student.iloc[:, 0:-1].values
y = student.iloc[:, -1].values
parameters = [{
    'max_depth': [7],
    'max_features': [0.75],
    'bootstrap': [False],
    'min_samples_leaf' : [5],
    'n_estimators': [1000]
}]
scaler = DummyUtilizer()
lf = LionForests(None, False, scaler, feature_names, class_names)
lf.fit(X, y, parameters)

discrete_features = ['Years_in_school','Number_of_courses_completed','Attending_class_per_week',
'Number_of_roomates']
categorical_features = ['Owns_a_car']
categorical_map = {'Owns_a_car':['dummy','No','Yes']}

if __name__ == "__main__":
    #description = "Hey there! Let’s predict the absence or presence of a serious killer disease in your heart. Are you in?"  # Statlog
    description = "Hey there! Let's predict if you are going to graduate at some point. Are you in?"  # Banknote
    #description = "Hey there! Wanna live the American Dream? Let's predict if you would make over 50K per year in the US. Are you in?"  # AC
    lfbot = LFBot(X, y, feature_names, categorical_features, class_names, parameters, description, lf,
                  discrete_features, categorical_map)
    lfbot.run()
