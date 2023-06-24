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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.decomposition import PCA

from preprocessing import *
from evaluation import *


TRAIN_FILE_PATH = "data/train2.csv"


# def find_best_number_of_neighbors(data, labels):
#     # create KNN classifier
#     knn = KNeighborsClassifier(n_neighbors=50)

#     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#     # fit the classifier to the data
#     knn.fit(X_train, y_train)

#     # perform 10-fold cross validation
#     scores = cross_val_score(knn, X_train, y_train, cv=10)

#     # find the mean accuracy across all 10 folds
#     mean_accuracy = scores.mean()

#     # find the optimal number of neighbors
#     optimal_k = np.argmax(scores) + 1

#     print("Optimal number of neighbors:", optimal_k)


@print_original_and_new_dataframes
def preform_pca(original_df, components):
    # create an instance of PCA

    # Create an instance of PCA with the number of components you want to keep
    pca = PCA(n_components=components)

    # create dataframe of all columns that their name starts with 'OHE'
    ohe_df = original_df.filter(regex='^OHE', axis=1)

    # Fit the data to the PCA model
    pca.fit(ohe_df)

    # Transform the data using the fitted model
    df_pca = pca.transform(ohe_df)

    # concat origial df while dropping the columns that were replaced by the PCA
    original_df = original_df.drop(original_df.filter(regex='^OHE', axis=1).columns, axis=1)
    out_df = pd.concat([original_df, pd.DataFrame(df_pca)], axis=1)

    return out_df



def debug_show_histogram(df):

    # copy the data frame
    df_copy = df.copy()

    # set all types to float64
    df_copy = df_copy.astype('float64')

    # show histogram of all numeric values
    df_copy.hist(figsize=(30, 30))
    plt.show()
    

def find_optimal_number_of_components_with_grid_search(X):
    pca = PCA()
    param_grid = {
        'n_components': range(5,25)
    }
    grid_search = GridSearchCV(pca, param_grid=param_grid)

    ohe_df = X.filter(regex='^OHE', axis=1)
    grid_search.fit(ohe_df)
    
    print(f"Best params for PCA: {grid_search.best_params_['n_components']}")



def main():
    df = pd.read_csv(TRAIN_FILE_PATH)
  
    labels = df['label']
    df.drop('label', axis=1, inplace=True)


    # preprocess the data frame
    df = preprocess_data(df)

    df = preform_pca(df, components=9) 

    #find_optimal_number_of_components_with_grid_search(df)

    data = df.values

    models = [svm.SVC(kernel='linear', degree=3)]

    # models = [KNeighborsClassifier(n_neighbors=9)] 
    # models = [RandomForestClassifier(n_estimators=300, max_depth=15, random_state=0)]

    for model in models:
        evaluate_with_train_test_split(model, data, labels)

    # evaluate_random_foreset(data, labels, None) # coloumns=df.columns)

    # find_best_params_for_rf(data, labels)




    


main()


"""
With PCA:
Accuracy score:  0.90425
F1 score:  0.9031768770540154


NO PCA:
Accuracy score:  0.9086666666666666
F1 score:  0.9079455736603393

"""
