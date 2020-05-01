from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

df = pd.read_csv('foss_para_arff_calibracao.csv', delimiter=';', decimal=',')
#df_validacao = pd.read_csv('foss_para_arff_validacao.csv', delimiter=';', decimal=',')


class Plsr():
    def __init__(self, dataset, components, cross_validation_type, split_for_validation=None, dataset_validation=None, scale=False, plsr_random_state=123):
        self.dataset = dataset
        self.components = components
        self.cross_validation_type = cross_validation_type
        self.split_for_validation = split_for_validation
        self.dataset_validation = dataset_validation
        self.scaleOpt = scale
        self.plsr_random_state = plsr_random_state

        self._xCal = pd.DataFrame()
        self._xVal = pd.DataFrame()
        self._yCal = pd.DataFrame()
        self._yVal = pd.DataFrame()

        self._cv = None

        x = dataset.iloc[:, 2:]
        y = dataset.iloc[:, 1]
        """
        This function receive by default a pandas Dataframe.
        """

        # checking if the parameters was inserted correctly


        # to do: determine priority between split original dataset or insert a new dataset for validation

        if not split_for_validation == None:
            if isinstance(split_for_validation, int) or isinstance(split_for_validation, float):
                self._xCal, self._xVal, self._yCal, self._yVal = train_test_split(x, y, test_size=split_for_validation, random_state=plsr_random_state)
            else:
                raise ValueError('split_for_validation need be a int or a float value')

        if not dataset_validation == None:
            if isinstance(dataset_validation, pd.DataFrame):
                self._xCal = dataset.iloc[:, 2:]
                self._yCal = dataset.iloc[:, 1]
                self._xVal = dataset_validation.iloc[:, 2:]
                self._yVal = dataset_validation.iloc[:, 1]
            else:
                raise ValueError("dataset_validation need be a pandas dataframe")

        if isinstance(cross_validation_type, str):
            if cross_validation_type == "loo":
                self._cv = LeaveOneOut()
        elif isinstance(cross_validation_type, int):
            self._cv = cross_validation_type
        else:
            raise ValueError("inser a valid value for define type of cross_validation. This value need be a int for k-fold method ou 'loo' for leave one out cross validation.")
    
    def train(self):
        """
        runs the plsr model with instance of PLSRegression from sklearn
        """        

        self.pls = PLSRegression(n_components=self.components, scale=self.scaleOpt)
        self.pls.fit(self._xCal, self._yCal)

        #pls.intercept_ = pls.y_mean_ - np.dot(pls.x_mean_, pls.coef_)

        # y_cv = cross_val_predict(pls, x, y, cv=_cv)
        # y_cv = [x[0] for x in y_cv]

        # y_c = pls.predict(x)
        # y_c = [x[0] for x in y_c]

        # y_val_pls = pls.predict(x_val)
        # y_val_pls = [x[0] for x in y_val_pls]

        # # linear model
        # print("R^2 de cal da regressao linear = {:.4f}".format(r2_linear_cal))
        # print("RMSE de cal da regressao linear = {:.4f}".format(rmse_linear_cal))
        # print("R^2 de cv da regressao linear = {:.4f}".format(r2_linear_cv))
        # print("RMSE de cv da regressao linear = {:.4f}".format(rmse_linear_cv))
        # print("R^2 de ve da regressao linear = {:.4f}".format(r2_linear_val))
        # print("RMSE de ve da regressao linear = {:.4f}\n".format(rmse_linear_val))

        # # pls
        # r2_cal = np.corrcoef(y.values, y_c)[0][1] ** 2
        # print("R^2 de calibração da PLS = {:.4f}".format(r2_cal))
        # rmse_cal = mean_squared_error(y.values, y_c)
        # print("RMSE da calibração da PLS = {:.4f}".format(rmse_cal))

        # #pls cv
        # r2_pls = np.corrcoef(y.values, y_cv)[0][1] ** 2
        # print("R^2 de cv da PLS = {:.4f}".format(r2_pls))
        # rmse_pls = mean_squared_error(y.values, y_cv)
        # print("RMSE de cv da PLS = {:.4f}".format(rmse_pls))

        # #pls ve
        # r2_pls_val = np.corrcoef(y_val, y_val_pls)[0][1] ** 2
        # print("R^2 de val da PLS = {:.4f}".format(r2_pls_val))
        # rmse_pls_pls = mean_squared_error(y_val, y_val_pls, squared=False)
        # print("RMSE de val da PLS = {:.4f}".format(rmse_pls_pls))
    def oi(self):
        print(self.pls.coef_)