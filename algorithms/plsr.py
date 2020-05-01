from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

df = pd.read_csv('foss_para_arff_calibracao.csv', delimiter=';', decimal=',')
df_validacao = pd.read_csv('foss_para_arff_validacao.csv', delimiter=';', decimal=',')

def plsr(dataset, components, kFold, split_for_validation=None, dataset_validation=None, scale=False):
    """
    This function receive by default a pandas Dataframe.
    """
    y = dataset.iloc[:,-1]
    x = dataset.iloc[:,:-1]
    y_val = df_validacao.iloc[:,-1]
    x_val = df_validacao.iloc[:,:-1]

    if not split_for_validation == None:
        if isinstance(split_for_validation, int) or isinstance(split_for_validation, float):
            pass
        else:
            raise ValueError('split_for_validation need be a int or a float value')

    if not dataset_validation == None:
        if isinstance(dataset_validation, pd.DataFrame):
            pass
        else:
            raise ValueError('dataset_validation need be a pandas dataframe')

    if kFold 
    loo = LeaveOneOut()

    # PLS
    pls = PLSRegression(n_components=components, scale=scale)
    pls.fit(x, y)
    #pls.intercept_ = pls.y_mean_ - np.dot(pls.x_mean_, pls.coef_)

    y_cv = cross_val_predict(pls, x, y, cv=loo)
    y_cv = [x[0] for x in y_cv]

    y_c = pls.predict(x)
    y_c = [x[0] for x in y_c]

    y_val_pls = pls.predict(x_val)
    y_val_pls = [x[0] for x in y_val_pls]

    # linear model
    print("R^2 de cal da regressao linear = {:.4f}".format(r2_linear_cal))
    print("RMSE de cal da regressao linear = {:.4f}".format(rmse_linear_cal))
    print("R^2 de cv da regressao linear = {:.4f}".format(r2_linear_cv))
    print("RMSE de cv da regressao linear = {:.4f}".format(rmse_linear_cv))
    print("R^2 de ve da regressao linear = {:.4f}".format(r2_linear_val))
    print("RMSE de ve da regressao linear = {:.4f}\n".format(rmse_linear_val))

    # pls
    r2_cal = np.corrcoef(y.values, y_c)[0][1] ** 2
    print("R^2 de calibração da PLS = {:.4f}".format(r2_cal))
    rmse_cal = mean_squared_error(y.values, y_c)
    print("RMSE da calibração da PLS = {:.4f}".format(rmse_cal))

    #pls cv
    r2_pls = np.corrcoef(y.values, y_cv)[0][1] ** 2
    print("R^2 de cv da PLS = {:.4f}".format(r2_pls))
    rmse_pls = mean_squared_error(y.values, y_cv)
    print("RMSE de cv da PLS = {:.4f}".format(rmse_pls))

    #pls ve
    r2_pls_val = np.corrcoef(y_val, y_val_pls)[0][1] ** 2
    print("R^2 de val da PLS = {:.4f}".format(r2_pls_val))
    rmse_pls_pls = mean_squared_error(y_val, y_val_pls, squared=False)
    print("RMSE de val da PLS = {:.4f}".format(rmse_pls_pls))