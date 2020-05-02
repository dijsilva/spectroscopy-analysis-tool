from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import numpy as np


def cross_validation(model, x, y, cv):
    y_cv = cross_val_predict(model, x, y, cv=cv)
    if len(y_cv.shape) == 2:
        y_cv = [i[0] for i in y_cv]

    r2 = np.corrcoef(y, y_cv)[0][1] ** 2
    rmse = mean_squared_error(y, y_cv, squared=False)

    predicted_values = np.array(y_cv)

    return (r2, rmse, predicted_values)


def external_validation(model, x, y):
    y_val = model.predict(x)

    if len(y_val.shape) == 2:
        y_val = [i[0] for i in y_val]

    r2_ve = np.corrcoef(y, y_val)[0][1] ** 2
    rmse_ve = mean_squared_error(y, y_val, squared=False)

    predicted_values = np.array(y_val)

    return (r2_ve, rmse_ve, predicted_values)