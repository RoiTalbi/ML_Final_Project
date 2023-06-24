
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import time 




# decorator to print the time it took to run the function
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time() - ts
        print(f"====> Time it took to run {method.__name__}: {te} seconds")
        return result
    return timed






def evaluate_model_with_cross_validation(model, data, labels):

    print(f"*******************  Evaluation model: {model.__class__.__name__} *******************")

    # Perform k-fold cross validation 
    scores = cross_validate(model, data, labels, cv=5, scoring = ['accuracy', 'f1'])
    print("------> Accuracy score: ", scores['test_accuracy'].mean())
    print("------>  F1 score: ", scores['test_f1'].mean())



def evaluate_with_train_test_split(model, data, labels):

    print(f"*******************  Evaluation model: {model.__class__.__name__} *******************")

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # train the model
    model.fit(X_train, y_train)

    # predict the test data
    y_pred = model.predict(X_test)

    # print("Classification report: \n", classification_report(y_test, y_pred))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("F1 score: ", f1_score(y_test, y_pred))



def find_best_params_for_rf(data, labels):
    
    model = RandomForestClassifier()

    print("*******************  Finding best params for Random Forest *******************")

    param_grid = {
    'n_estimators': [300, 400, 500],
    'max_depth': [15, 20, 30],
    'n_jobs': [-1]
    }
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("Best score: ")
    print(grid_search.best_score_)

    print("Best params: ")
    print(grid_search.best_params_)





@timeit
def evaluate_random_foreset(data, labels, coloumns):

    # create an instance of the model
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)

    # evaluate the model  
    evaluate_with_train_test_split(rf, data, labels)

    if coloumns is not None:
        print("Feature importance: ")
        importances = pd.Series(rf.feature_importances_, index=coloumns)
        importances.nlargest(30).plot(kind='barh')
        plt.show()


def evaluate_model(model, data, labels):

    print(f"*******************  Evaluation model: {model.__class__.__name__} *******************")

    evaluate_model_with_cross_validation(model, data, labels)



