from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

from utils import cross_validation, external_validation

class RandomForest():
    def __init__(self, dataset, estimators=100, cross_validation_type='loo', split_for_validation=None, dataset_validation=None, rf_random_state=1, 
                max_features_rf='auto', rf_max_depth=None, rf_min_samples_leaf=1, rf_min_samples_split=2, rf_Bootstrap=True, rf_oob_score=False):
        self.dataset = dataset
        self.n_estimators = estimators
        self.cross_validation_type = cross_validation_type
        self.split_for_validation = split_for_validation
        self.dataset_validation = dataset_validation
        self.rf_random_state = rf_random_state
        self.max_features = max_features_rf
        self.max_depth = rf_max_depth
        self.min_samples_leaf = rf_min_samples_leaf
        self.min_samples_split = rf_min_samples_split
        self.bootstrap = rf_Bootstrap
        self.oob_score = rf_oob_score

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
            cv = KFold(cross_validation_type, shuffle=True, random_state=self.rf_random_state)
            self._cv = cv
        else:
            raise ValueError("The cross_validation_type should be a positive integer for k-fold method ou 'loo' for leave one out cross validation.")
    

    def search_hyperparameters(self, estimators=[100, 1010], max_features=['sqrt'], max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], 
                               bootstrap=[True], n_processors=1, verbose=0, oob_score=[False], scoring='neg_root_mean_squared_error'):
        
        stop_value = lambda list_of_values: 10 if (len(list_of_values) < 3) else list_of_values[2]
        n_estimators = [int(x) for x in np.arange(start = estimators[0], stop = estimators[1], step = stop_value(estimators))]
        if None not in max_depth:
            max_depth = [int(x) for x in np.arange(max_depth[0], max_depth[1], step = stop_value(max_depth))]
            max_depth.append(None)

        random_grid = { "n_estimators": n_estimators,
                        "max_features": max_features,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "bootstrap": bootstrap,
                        "oob_score": oob_score                         
                       }
    
        rf = RandomForestRegressor()

        rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = self._cv, n_jobs = n_processors, verbose=verbose, scoring=scoring)
        
        if verbose == 0:
            print('Running...')
        
        rf_random.fit(self._xCal, self._yCal)

        get_params = lambda dict_params, param, default_params: dict_params[param] if (param in dict_params) else default_params
        
        self._best_params = rf_random.best_params_
        self.n_estimators = get_params(rf_random.best_params_, 'n_estimators', self.n_estimators)
        self.max_features = get_params(rf_random.best_params_, 'max_features', self.max_features)
        self.max_depth = get_params(rf_random.best_params_, 'max_depth', self.max_depth)
        self.min_samples_leaf = get_params(rf_random.best_params_, 'min_samples_leaf', self.min_samples_leaf)
        self.min_samples_split = get_params(rf_random.best_params_, 'min_samples_split', self.min_samples_split)
        self.bootstrap = get_params(rf_random.best_params_, 'bootstrap', self.bootstrap)
        self.oob_score = get_params(rf_random.best_params_, 'oob_score', self.oob_score)

    def calibrate(self):
        
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.rf_random_state, 
                                         max_features=self.max_features, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, 
                                         min_samples_split=self.min_samples_split, bootstrap=self.bootstrap, oob_score=self.oob_score)

        self.model.fit(self._xCal, self._yCal)

        y_cal_predict = self.model.predict(self._xCal)

        r_correlation = np.corrcoef(self._yCal, y_cal_predict)[0][1]
        r2_cal = self.model.score(self._xCal, self._yCal)
        rmse = mean_squared_error(self._yCal, y_cal_predict, squared=False)

        nsamples = self._xCal.shape[0]

        calibration_metrics = {'n_samples': nsamples, 'R': r_correlation, 'R2': r2_cal, 'RMSE': rmse}

        self.metrics['calibration'] = calibration_metrics  
    


    def cross_validate(self):
        
        r_correlation, r2_cv, rmse_cv, predicted_values = cross_validation(self.model, self._xCal, self._yCal, self._cv, correlation_based=False)

        method = 'Leave One Out'
        if isinstance(self._cv, KFold):
            method = "{}-fold".format(self._cv.n_splits)
        
        cross_validation_metrics = {'R': r_correlation, 'R2': r2_cv, 'RMSE': rmse_cv, 'method': method, 'predicted_values': predicted_values }
        
        self.metrics['cross_validation'] = cross_validation_metrics
    
    def validate(self):

        r_correlation, r2_ve, rmse_ve, predicted_values = external_validation(self.model, self._xVal, self._yVal, correlation_based=False)

        nsamples = self._xVal.shape[0]
        validation = {'R': r_correlation, 'R2': r2_ve, 'RMSE': rmse_ve, 'n_samples': nsamples, 'predicted_values': predicted_values}

        self.metrics['validation'] = validation

    def create_model(self):

        self.calibrate()
        self.cross_validate()
        self.validate()