
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report, confusion_matrix





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

    print("Classification report: \n", classification_report(y_test, y_pred))
    print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))




def evaluate_model(model, data, labels):

    print(f"*******************  Evaluation model: {model.__class__.__name__} *******************")
    evaluate_model_with_cross_validation(model, data, labels)
