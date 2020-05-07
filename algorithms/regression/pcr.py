import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
from utils import external_validation, cross_validation

class PCR():
    def __init__(self, dataset, components=2, cross_validation_type='loo', split_for_validation=None, dataset_validation=None, fit_intercept_regression=True, pcr_random_state=123):
        self.dataset = dataset
        self.components = components
        self.cross_validation_type = cross_validation_type
        self.split_for_validation = split_for_validation
        self.dataset_validation = dataset_validation
        self.fit_intercept_regression = fit_intercept_regression
        self.pcr_random_state = pcr_random_state

        self._xCal = pd.DataFrame()
        self._xVal = pd.DataFrame()
        self._yCal = pd.DataFrame()
        self._yVal = pd.DataFrame()

        self._cv = None

        self.metrics = {}

        # checking if the parameters was inserted correctly

        if not isinstance(self.dataset, pd.DataFrame):
            raise ValueError('The dataset should be a pd.DataFrame.')

        if (self.dataset_validation is None) and (self.split_for_validation is None):
            raise ValueError('Should be defined the samples for validation or size of test size for split the dataset.')

        x = dataset.iloc[:, 2:]
        y = dataset.iloc[:, 1]
        
        if (not self.split_for_validation is None) and (self.dataset_validation is None):
            if isinstance(self.split_for_validation, float):
                self._xCal, self._xVal, self._yCal, self._yVal = train_test_split(x, y, test_size=split_for_validation, random_state=pcr_random_state)
            else:
                raise ValueError('split_for_validation need be a float value between 0 and 1')

        if not self.dataset_validation is None:
            if isinstance(self.dataset_validation, pd.DataFrame):
                self._xCal = self.dataset.iloc[:, 2:]
                self._yCal = self.dataset.iloc[:, 1]
                self._xVal = self.dataset_validation.iloc[:, 2:]
                self._yVal = self.dataset_validation.iloc[:, 1]
            else:
                raise ValueError("dataset_validation need be a pandas dataframe")


        if isinstance(cross_validation_type, str):
            if cross_validation_type == "loo":
                self._cv = LeaveOneOut()
            elif (type(cross_validation_type) in [int]) and (cross_validation_type > 0):
                self._cv = cross_validation_type
            else:
                raise ValueError("The cross_validation_type should be a positive integer for k-fold method ou 'loo' for leave one out cross validation.")

        if self.fit_intercept_regression not in [True, False]:
            raise ValueError('The scale option should be a boolean.')
    
    
    def calibrate(self):

        # reduce dimensionality of self._xCal with pca and store the components in self._Xreduced
        self._pca = PCA(n_components=self.components, tol=0.0000000001, random_state=self.pcr_random_state, svd_solver="full")
        self._linear_regression = LinearRegression(fit_intercept=self.fit_intercept_regression)


        self._Xreduced = self._pca.fit_transform(scale(self._xCal, axis=1))

        self._linear_regression.fit(self._Xreduced, self._yCal)

        y_cal_predict = self._linear_regression.predict(self._Xreduced)

        r2_cal = np.corrcoef(y_cal_predict, self._yCal)[0][1] ** 2
        rmse = mean_squared_error(self._yCal, y_cal_predict, squared=False)

        nsamples = self._Xreduced.shape[0]

        calibration_metrics = {'n_samples': nsamples, 'R2': r2_cal, 'RMSE': rmse}

        self.metrics['calibration'] = calibration_metrics

    
    def cross_validate(self):

        r2_cv, rmse_cv, predicted_values = cross_validation(self._linear_regression, self._Xreduced, self._yCal, cv=self._cv)

        method = 'LOO'
        if isinstance(self._cv, int):
            method = "{}-fold".format(self._cv)


        cross_validation_metrics = {'R2': r2_cv, 'RMSE': rmse_cv, 'method': method, 'predicted_values': predicted_values}

        self.metrics['cross_validation'] = cross_validation_metrics
    
    
    def validate(self):

        self._pca_val = PCA(n_components=self.components, tol=0.0000000001, random_state=self.pcr_random_state, svd_solver="full")

        self._XValReduced = self._pca_val.fit_transform(scale(self._xVal, axis=1))
        
        r2_ve, rmse_ve, predicted_values = external_validation(self._linear_regression, self._XValReduced, self._yVal)
        
        nsamples = self._xVal.shape[0]
        validation = {'R2': r2_ve, 'RMSE': rmse_ve, 'n_samples': nsamples, 'predicted_values': predicted_values}

        self.metrics['validation'] = validation
    
   
    def get_coefs(self, get_intercept=False):
        
        # return a array with coefficientes. If get_intercept == True, then intercept is calculated 
        # an insert in coefs array at index 0
        
        coefs = self._linear_regression.coef_
        if get_intercept == True:
            coefs = np.insert(coefs, 0, self._linear_regression.intercept_)    

        return coefs
    

    def create_model(self):
        
        # this function should be used to calibrate, cross-validate and validate with one command

        self.calibrate()
        self.cross_validate()
        self.validate()