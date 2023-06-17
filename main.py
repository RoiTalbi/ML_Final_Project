import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

from preprocessing import *
from evaluation import *


TRAIN_FILE_PATH = "data/train2.csv"


def find_best_number_of_neighbors(data, labels):
    # create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=50)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # fit the classifier to the data
    knn.fit(X_train, y_train)

    # perform 10-fold cross validation
    scores = cross_val_score(knn, X_train, y_train, cv=10)

    # find the mean accuracy across all 10 folds
    mean_accuracy = scores.mean()

    # find the optimal number of neighbors
    optimal_k = np.argmax(scores) + 1

    print("Optimal number of neighbors:", optimal_k)




def main():
    df = pd.read_csv(TRAIN_FILE_PATH)
  
    labels = df['label']
    df.drop('label', axis=1, inplace=True)

    # preprocess the data frame
    df = preprocess_data(df)
   
    data = df.values

    # plt show the variance of each column
    #df.var().plot(kind='bar')
    #plt.show()


    # print df colomns
    print(f"=====> Columns: {df.columns}")

    # get the histogram of each column
    # df.hist(figsize=(20,20))
    # plt.show()

    # models = []
    models = [KNeighborsClassifier(n_neighbors=9)] #, RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)]
    for model in models:
         evaluate_with_train_test_split(model, data, labels)


    


main()







# ---------------------------------------- OLD CODE ----------------------------------------

def object_to_numeric(obj):
    
    # convert the obj to string
    obj_str = str(obj)

    # hash obj_str to int value with sha256
    hashed = hash(obj_str)

    return hashed % 1000

