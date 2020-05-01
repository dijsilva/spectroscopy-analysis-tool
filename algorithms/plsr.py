from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

#df = pd.read_csv('/home/dsilva/foss_para_arff_calibracao.csv', delimiter=';', decimal=',')
#df_validacao = pd.read_csv('/home/dsilva/foss_para_arff_validacao.csv', delimiter=';', decimal=',')


class PLSR():
    """
        This class receive by default a pandas Dataframe and some params for perform a PLS regression.
        
        The params are:
            - components: number of components in pls regression
            - cross_validation_type: type of cross-validation that will be performed. Insert int to use 
            k-fold strategy and 'loo' to use leave one out strategy. Deafult is 'loo'.
            - split_for_validation: should be a float between 0 and 1. If informed, this represents a size of
            dataset that will be used as test samples.
            - dataset_validation: If informed, should be a dataframe with sample that will be used for validate model
            - scale: if true, then the data is scaled
            - plsr_random_state: is the seed for random number generetor. Its used for split dataset
        
        If split_for_validation and dataset_validation are both informed, then only dataset_validation is considered.
    """

    def __init__(self, dataset, components=2, cross_validation_type='loo', split_for_validation=None, dataset_validation=None, scale=False, plsr_random_state=123):
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
        self._predictions = pd.DataFrame()

        self._cv = None

        x = dataset.iloc[:, 2:]
        y = dataset.iloc[:, 1]

        # checking if the parameters was inserted correctly
        if (not self.split_for_validation == None) and (self.dataset_validation == None):
            if isinstance(self.split_for_validation, float):
                self._xCal, self._xVal, self._yCal, self._yVal = train_test_split(x, y, test_size=split_for_validation, random_state=plsr_random_state)
            else:
                raise ValueError('split_for_validation need be a float value between 0 and 1')

        if not self.dataset_validation == None:
            if isinstance(self.dataset_validation, pd.DataFrame):
                self._xCal = self.dataset.iloc[:, 2:]
                self._yCal = self.dataset.iloc[:, 1]
                self._xVal = self.dataset_validation.iloc[:, 2:]
                self._yVal = self.dataset_validation.iloc[:, 1]
            else:
                raise ValueError("dataset_validation need be a pandas dataframe")

        if (self.dataset_validation == None) and (self.split_for_validation == None):
            raise ValueError('Should be defined or informed the dataset used for validate model.')

        if isinstance(cross_validation_type, str):
            if cross_validation_type == "loo":
                self._cv = LeaveOneOut()
        elif isinstance(cross_validation_type, int):
            self._cv = cross_validation_type
        else:
            raise ValueError("inser a valid value for define type of cross_validation. This value need be a int for k-fold method ou 'loo' for leave one out cross validation.")
    
    
    def calibrate(self):
        """
        runs the plsr model with instance of PLSRegression from sklearn
        """        

        self.pls = PLSRegression(n_components=self.components, scale=self.scaleOpt)
        self.pls.fit(self._xCal, self._yCal)

        x_cal_predict = self.pls.predict(self._xCal)

        x_cal_predict = [i[0] for i in x_cal_predict]
        r2_cal = np.corrcoef(x_cal_predict, self._yCal)[0][1] ** 2
        rmse = mean_squared_error(self._yCal, x_cal_predict, squared=False)

        nsamples = self._xCal.shape[0]
        print(f"MODEL CALIBRATED with {nsamples} samples and {self.components} components")
        print(f"Coefficiente of determination = {r2_cal:.3f}")
        print(f"RMSE = {rmse:.3f}")
    
    
    def cross_validate(self):
        
        y_cv = cross_val_predict(self.pls, self._xCal, self._yCal, cv=self._cv)
        y_cv = [i[0] for i in y_cv]

        r2_cv = np.corrcoef(self._yCal, y_cv)[0][1] ** 2
        rmse_cv = mean_squared_error(self._yCal, y_cv, squared=False)

        print("CROSS-VALIDATION COMPLETED:")
        print(f"Coefficient of determination = {r2_cv:.3f}")
        print(f"RMSE = {rmse_cv:.3f}")
    

    def validate(self):

        y_val = self.pls.predict(self._xVal)
        y_val = [i[0] for i in y_val]

        r2_ve = np.corrcoef(self._yVal, y_val)[0][1] ** 2
        rmse_ve = mean_squared_error(self._yVal, y_val, squared=False)

        print("VALIDATION COMPLETED:")
        print(f"Coefficient of determination = {r2_ve:.3f}")
        print(f"RMSE = {rmse_ve:.3f}")
    

    def get_coefs(self, get_intercept=False):
        """
        return a array with coefficientes. If get_intercept == True, then intercept is calculated an insert in coefs array at index 0
        """
        coefs = np.array([coef[0] for coef in self.pls.coef_])
        if get_intercept == True:
            self.pls.intercept_ = self.pls.y_mean_ - np.dot(self.pls.x_mean_, self.pls.coef_)
            coefs = np.insert(coefs, 0, self.pls.intercept_)    

        return coefs
