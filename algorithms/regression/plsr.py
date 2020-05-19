from sklearn.model_selection import cross_val_predict, LeaveOneOut, train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from utils import external_validation, cross_validation

import pandas as pd
import numpy as np

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

    def __init__(self, dataset, components=2, cross_validation_type='loo', split_for_validation=None, dataset_validation=None, scale=True, plsr_random_state=123):
        self.dataset = dataset
        self.components = components
        self.cross_validation_type = cross_validation_type
        self.split_for_validation = split_for_validation
        self.dataset_validation = dataset_validation
        self.scaleOpt = scale
        self.modelr_random_state = plsr_random_state

        self._xCal = pd.DataFrame()
        self._xVal = pd.DataFrame()
        self._yCal = pd.DataFrame()
        self._yVal = pd.DataFrame()

        self._cv = None

        self.metrics = {}

        # checking if the parameters was inserted correctly

        if not isinstance(self.dataset, pd.DataFrame):
            raise ValueError('The dataset should be a pd.DataFrame.')

        if type(self.components) not in [int]:
            raise ValueError('components should be a integer')

        if (self.dataset_validation is None) and (self.split_for_validation is None):
            raise ValueError('Should be defined the samples for validation or size of test size for split the dataset.')
        
        if (not self.split_for_validation is None) and (self.dataset_validation is None):
            if self.split_for_validation == 'all':
                self._xCal = self.dataset.iloc[:, 2:]
                self._yCal = self.dataset.iloc[:, 1]
            elif isinstance(self.split_for_validation, float):
                self._xCal, self._xVal, self._yCal, self._yVal = train_test_split(self.dataset.iloc[:, 2:], self.dataset.iloc[:, 1], test_size=split_for_validation, random_state=self.rf_random_state)
            else:
                raise ValueError("split_for_validation need be a float value between 0 and 1 for split dataset. Use 'all' for calibrate with all samples of dataset.")

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

        if self.scaleOpt not in [True, False]:
            raise ValueError('The scale option should be a boolean.')
    
    def search_hyperparameters(self, components=[1, 21], n_processors=1, verbose=0, scoring='neg_root_mean_squared_error'):
        
        step_value = lambda list_of_values: 1 if (len(list_of_values) < 3) else list_of_values[2]
        components = [int(x) for x in np.arange(start = components[0], stop = components[1], step = step_value(components))]
        
        grid = { "n_components": components }

        pls = PLSRegression(scale=self.scaleOpt)
        pls_grid_search = GridSearchCV(estimator = pls, param_grid = grid, cv = self._cv, n_jobs = n_processors, verbose=verbose, scoring=scoring)

        pls_grid_search.fit(self._xCal, self._yCal)

        get_params = lambda dict_params, param, default_params: dict_params[param] if (param in dict_params) else default_params
        
        self._best_params = pls_grid_search.best_params_
        self.components = get_params(pls_grid_search.best_params_, 'n_components', self.components)
    
    
    def calibrate(self):
        """
        runs the plsr model with instance of PLSRegression from sklearn
        """        

        self.model = PLSRegression(n_components=self.components, scale=self.scaleOpt)
        self.model.fit(self._xCal, self._yCal)

        y_cal_predict = self.model.predict(self._xCal)
        y_cal_predict = [i[0] for i in y_cal_predict]
        
        r_correlation = np.corrcoef(y_cal_predict, self._yCal)[0][1]
        
        r2_cal = self.model.score(self._xCal, self._yCal)

        rmse = mean_squared_error(self._yCal, y_cal_predict, squared=False)

        nsamples = self._xCal.shape[0]

        calibration_metrics = {'n_samples': nsamples, 'R': r_correlation, 'R2': r2_cal, 'RMSE': rmse}

        self.metrics['calibration'] = calibration_metrics        
    
    
    def cross_validate(self):
        
        r_correlation, r2_cv, rmse_cv, predicted_values = cross_validation(self.model, self._xCal, self._yCal, cv=self._cv)

        method = 'LOO'
        if isinstance(self._cv, int):
            method = "{}-fold".format(self._cv)

        cross_validation_metrics = {'R': r_correlation, 'R2': r2_cv, 'RMSE': rmse_cv, 'method': method, 'predicted_values': predicted_values }

        self.metrics['cross_validation'] = cross_validation_metrics
    

    def validate(self):

        r_correlation, r2_ve, rmse_ve, predicted_values = external_validation(self.model, self._xVal, self._yVal)

        nsamples = self._xVal.shape[0]
        validation = {'R': r_correlation, 'R2': r2_ve, 'RMSE': rmse_ve, 'n_samples': nsamples, 'predicted_values': predicted_values}

        self.metrics['validation'] = validation
    

    def get_coefs(self, get_intercept=False):
        
        # return a array with coefficientes. If get_intercept == True, then intercept is calculated 
        # an insert in coefs array at index 0
        
        coefs = np.array([coef[0] for coef in self.model.coef_])
        if get_intercept == True:
            self.model._intercept = self.model.y_mean_ - np.dot(self.model.x_mean_, self.model.coef_)
            coefs = np.insert(coefs, 0, self.model._intercept)    

        return coefs
    
    def create_model(self):
        
        # this function should be used to calibrate, cross-validate and validate with one command

        self.calibrate()
        self.cross_validate()
        self.validate()

