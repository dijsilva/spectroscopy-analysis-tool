from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, r2_score
import numpy as np
import pandas as pd


def cross_validation(model, x, y, cv, correlation_based=True):
    y_cv = cross_val_predict(model, x, y, cv=cv)
    if len(y_cv.shape) == 2:
        y_cv = [i[0] for i in y_cv]
    
    bias = (y_cv - y).sum() / y.shape[0]

    r_correlation = np.corrcoef(y, y_cv)[0][1]
    
    if correlation_based == True:
        r2 = r_correlation ** 2
    else:
        r2 = r2_score(y, y_cv)
    rmse = mean_squared_error(y, y_cv, squared=False)

    predicted_values = np.array(y_cv)

    return (r_correlation, r2, rmse, bias, predicted_values)

def classifier_cross_validation(model, x, y, cv):
    y_cv = cross_val_predict(model, x, y, cv=cv)

    accuracy = accuracy_score(y, y_cv)
    matrix = confusion_matrix(y, y_cv)

    cm = pd.DataFrame(matrix)

    index_columns = y.value_counts().sort_index(axis=0).index
    cm.index = index_columns
    cm.columns = index_columns
    
    predicted_values = np.array(y_cv)

    return (accuracy, cm, predicted_values)


def external_validation(model, x, y, correlation_based=True):
    y_val = model.predict(x)

    if len(y_val.shape) == 2:
        y_val = [i[0] for i in y_val]
    
    r_correlation = np.corrcoef(y, y_val)[0][1]

    bias = (y_val - y).sum() / y.shape[0]
    
    if correlation_based == True:
        r2_ve = r_correlation ** 2
    else:
        r2_ve = r2_score(y, y_val)
    rmse_ve = mean_squared_error(y, y_val, squared=False)

    predicted_values = np.array(y_val)

    return (r_correlation, r2_ve, rmse_ve, bias, predicted_values)


def classifier_external_validation(model, x, y):

    predictions = model.predict(x)

    accuracy = accuracy_score(y, predictions)
    matrix = confusion_matrix(y, predictions, labels=model.classes_)

    cm = pd.DataFrame(matrix)

    cm.index = model.classes_
    cm.columns = model.classes_

    predicted_values = np.array(predictions)

    return (accuracy, cm, predicted_values)



