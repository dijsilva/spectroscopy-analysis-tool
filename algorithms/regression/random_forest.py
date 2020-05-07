from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, LeaveOneOut, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

from utils import cross_validation, external_validation

class RandomForest():
    def __init__(self, dataset, estimators=100, cross_validation_type='loo', split_for_validation=None, dataset_validation=None, rf_random_state=1):
        self.dataset = dataset
        self.estimators = estimators
        self.cross_validation_type = cross_validation_type
        self.split_for_validation = split_for_validation
        self.dataset_validation = dataset_validation
        self._rf_random_state = rf_random_state

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
                self._xCal, self._xVal, self._yCal, self._yVal = train_test_split(x, y, test_size=split_for_validation, random_state=plsr_random_state)
            else:
                raise ValueError('split_for_validation need be a float value between 0 and 1')


        if not self.dataset_validation is None:
            if isinstance(self.dataset_validation, pd.DataFrame):
                self._xCal = self.dataset.iloc[:, 2:]
                self._yCal = self.dataset.iloc[:, 1]
                self._xVal = self.dataset_validation.iloc[:, 2:]
                self._yVal = self.dataset_validation.iloc[:, 1]
            else:
                raise ValueError("dataset_validation need be a pd.DataFrame")


        if isinstance(cross_validation_type, str):
            if cross_validation_type == "loo":
                self._cv = LeaveOneOut()
        elif (type(cross_validation_type) in [int]) and (cross_validation_type > 0):
            self._cv = cross_validation_type
        else:
            raise ValueError("The cross_validation_type should be a positive integer for k-fold method ou 'loo' for leave one out cross validation.")
        

    def calibrate(self):
        
        self._rf = RandomForestRegressor(n_estimators=self.estimators, random_state=self._rf_random_state)

        self._rf.fit(self._xCal, self._yCal)

        y_cal_predict = self._rf.predict(self._xCal)
        
        r2_cal = np.corrcoef(y_cal_predict, self._yCal)[0][1] ** 2
        rmse = mean_squared_error(self._yCal, y_cal_predict, squared=False)

        nsamples = self._xCal.shape[0]

        calibration_metrics = {'n_samples': nsamples, 'R2': r2_cal, 'RMSE': rmse}

        self.metrics['calibration'] = calibration_metrics  
    

    def cross_validate(self):
        
        r2_cv, rmse_cv, predicted_values = cross_validation(self._rf, self._xCal, self._yCal, self._cv)

        method = 'LOO'
        if isinstance(self._cv, int):
            method = "{}-fold".format(self._cv)
        
        cross_validation_metrics = {'R2': r2_cv, 'RMSE': rmse_cv, 'method': method, 'predicted_values': predicted_values }
        
        self.metrics['cross_validation'] = cross_validation_metrics
    
    def validate(self):

        r2_ve, rmse_ve, predicted_values = external_validation(self._rf, self._xVal, self._yVal)

        nsamples = self._xVal.shape[0]
        validation = {'R2': r2_ve, 'RMSE': rmse_ve, 'n_samples': nsamples, 'predicted_values': predicted_values}

        self.metrics['validation'] = validation

    def create_model(self):

        self.calibrate()
        self.cross_validate()
        self.validate()
