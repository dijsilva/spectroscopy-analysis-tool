from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

df = pd.read_csv('/home/dsilva/home_office/algoritmo_mateus_arff/foss_para_arff_calibracao.csv', delimiter=';', decimal=',')
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
    
    def run(self):
        """
        runs the plsr model with instance of PLSRegression from sklearn
        """        

        self.pls = PLSRegression(n_components=self.components, scale=self.scaleOpt)
        self.pls.fit(self._xCal, self._yCal)

  
    def get_coefs(self, get_intercept=False):
        if get_intercept == True:
            self.pls.intercept_ = self.pls.y_mean_ - np.dot(self.pls.x_mean_, self.pls.coef_)
            coefs = np.array([coef[0] for coef in self.pls.coef_])
            coefs = np.insert(coefs, 0, self.pls.intercept_)

            return coefs
        else:
            coefs = np.array([coef[0] for coef in self.pls.coef_])

            return coefs
