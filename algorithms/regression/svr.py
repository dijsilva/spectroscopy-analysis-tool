from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut, train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

from utils import cross_validation, external_validation

class SVMRegression():
    def __init__(self, dataset, type_of_kernel = 'rbf', cross_validation_type = 'loo', split_for_validation = None, dataset_validation = None, svm_random_state = 1, 
                svm_degree=3, svm_gamma='scale', svm_coef=0.0, svm_tol=1e-10, svm_epsilon=0.1, svm_max_iter=-1):
        self.dataset = dataset
        self.kernel = type_of_kernel
        self.cross_validation_type = cross_validation_type
        self.split_for_validation = split_for_validation
        self.dataset_validation = dataset_validation
        self.svm_random_state = svm_random_state
        
        self.degree = svm_degree
        self.gamma = svm_gamma
        self.coef0 = svm_coef
        self.tol = svm_tol
        self.epsilon = svm_epsilon
        self.max_iter = svm_max_iter

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

        # x = dataset.iloc[:, 2:]
        # y = dataset.iloc[:, 1]
        
        if (not self.split_for_validation is None) and (self.dataset_validation is None):
            if self.split_for_validation == 'all':
                self._xCal = self.dataset.iloc[:, 2:]
                self._yCal = self.dataset.iloc[:, 1]
            elif isinstance(self.split_for_validation, float):
                self._xCal, self._xVal, self._yCal, self._yVal = train_test_split(self.dataset.iloc[:, 2:], self.dataset.iloc[:, 1], test_size=split_for_validation, random_state=self.svm_random_state)
            else:
                raise ValueError("split_for_validation need be a float value between 0 and 1 for split dataset. Use 1 for calibrate with all samples of dataset.")


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
            cv = KFold(cross_validation_type, shuffle=True, random_state=self.svm_random_state)
            self._cv = cv
        else:
            raise ValueError("The cross_validation_type should be a positive integer for k-fold method ou 'loo' for leave one out cross validation.")
    

    def search_hyperparameters(self, kernel = ['rbf'], degree = [ 3 ], gamma=[ 'scale' ], coef0=[ 0.0, 0.1 ], epsilon=[ 0.1, 2.0 ], 
                               tol = [1e-3, 1e-10], max_iter = [ -1 ], n_processors = 1, verbose = 0, 
                               scoring = 'neg_root_mean_squared_error'):
        
        step_value = lambda list_of_values: 0.5 if (len(list_of_values) < 3) else list_of_values[2]
        epsilon = [round(x, 3) for x in np.arange(start = epsilon[0], stop = epsilon[1], step = step_value(epsilon))]
        coef0 = [round(x, 3) for x in np.arange(start = coef0[0], stop = coef0[1], step = step_value(coef0))]

        random_grid = { "kernel": kernel,
                        "degree": degree,
                        "gamma": gamma,
                        "coef0": coef0,
                        "epsilon": epsilon,
                        "max_iter": max_iter,
                        "tol": tol
                       }
    
        svm_regression = SVR()

        svm_regresion_grid = GridSearchCV(estimator = svm_regression, param_grid = random_grid, cv = self._cv, n_jobs = n_processors, verbose=verbose, scoring=scoring)
        svm_regresion_grid.fit(self._xCal, self._yCal)

        get_params = lambda dict_params, param, default_params: dict_params[param] if (param in dict_params) else default_params
        
        self._best_params = svm_regresion_grid.best_params_
        self.kernel = get_params(svm_regresion_grid.best_params_, 'kernel', self.kernel)
        self.degree = get_params(svm_regresion_grid.best_params_, 'degree', self.degree)
        self.gamma = get_params(svm_regresion_grid.best_params_, 'gamma', self.gamma)
        self.coef0 = get_params(svm_regresion_grid.best_params_, 'coef0', self.coef0)
        self.tol = get_params(svm_regresion_grid.best_params_, 'tol', self.tol)
        self.epsilon = get_params(svm_regresion_grid.best_params_, 'epsilon', self.epsilon)
        self.max_iter = get_params(svm_regresion_grid.best_params_, 'max_iter', self.max_iter)

    def calibrate(self):
        
        self.model = SVR(kernel = self.kernel, degree = self.degree, gamma = self.gamma, coef0 = self.coef0, tol = self.tol, 
                         epsilon = self.epsilon, max_iter = self.max_iter)

        self.model.fit(self._xCal, self._yCal)

        y_cal_predict = self.model.predict(self._xCal)
        r_correlation = np.corrcoef(self._yCal, y_cal_predict)[0][1]
        r2_cal = self.model.score(self._xCal, self._yCal)
        rmse = mean_squared_error(self._yCal, y_cal_predict, squared=False)

        nsamples = self._xCal.shape[0]

        calibration_metrics = {'n_samples': nsamples, 'R': r_correlation, 'R2': r2_cal, 'RMSE': rmse}

        self.metrics['calibration'] = calibration_metrics  
    


    def cross_validate(self):
        
        r_correlation, r2_cv, rmse_cv, bias, predicted_values = cross_validation(self.model, self._xCal, self._yCal, self._cv, correlation_based=False)

        method = 'Leave One Out'
        if isinstance(self._cv, KFold):
            method = "{}-fold".format(self._cv.n_splits)
        
        cross_validation_metrics = {'R': r_correlation, 'R2': r2_cv, 'RMSE': rmse_cv, 'bias': bias, 'method': method, 'predicted_values': predicted_values }
        
        self.metrics['cross_validation'] = cross_validation_metrics
    
    def validate(self):

        r_correlation, r2_ve, rmse_ve, bias, predicted_values = external_validation(self.model, self._xVal, self._yVal, correlation_based=False)

        nsamples = self._xVal.shape[0]
        validation = {'R': r_correlation, 'R2': r2_ve, 'RMSE': rmse_ve, 'bias': bias, 'n_samples': nsamples, 'predicted_values': predicted_values}

        self.metrics['validation'] = validation

    def create_model(self):

        self.calibrate()
        self.cross_validate()
        self.validate()