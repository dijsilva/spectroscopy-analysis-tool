from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, LeaveOneOut, train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import os

from utils import cross_validation, external_validation

class RandomForest():
    def __init__(self, dataset, estimators=100, cross_validation_type='loo', split_for_validation=None, dataset_validation=None, rf_random_state=1, 
                max_features_rf='auto', rf_max_depth=None, rf_min_samples_leaf=1, rf_min_samples_split=2, rf_Bootstrap=True, rf_oob_score=False):
        self.dataset = dataset
        self.n_estimators = estimators
        self.cross_validation_type = cross_validation_type
        self.split_for_validation = split_for_validation
        self.dataset_validation = dataset_validation
        self._rf_random_state = rf_random_state
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
                self._xCal, self._xVal, self._yCal, self._yVal = train_test_split(self.dataset.iloc[:, 2:], self.dataset.iloc[:, 1], test_size=split_for_validation, random_state=lda_random_state)
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
            cv = KFold(cross_validation_type, shuffle=True, random_state=self._rf_random_state)
            self._cv = cv
        else:
            raise ValueError("The cross_validation_type should be a positive integer for k-fold method ou 'loo' for leave one out cross validation.")
    

    def search_hyperparameters(self, estimators=[100, 1010], max_features=['sqrt'], max_depth=[10, 110], min_samples_split=[2], min_samples_leaf=[1], 
                               bootstrap=[True], n_processors=1, verbose=0, oob_score=[False]):
        
        stop_value = lambda list_of_values: 10 if (len(list_of_values) < 3) else list_of_values[2]
        
        n_estimators = [int(x) for x in np.arange(start = estimators[0], stop = estimators[1], step = stop_value(estimators))]
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

        rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = self._cv, n_jobs = n_processors, verbose=verbose)
        
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
        
        self._rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self._rf_random_state, 
                                         max_features=self.max_features, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, 
                                         min_samples_split=self.min_samples_split, bootstrap=self.bootstrap, oob_score=self.oob_score)

        self._rf.fit(self._xCal, self._yCal)

        y_cal_predict = self._rf.predict(self._xCal)

        r_correlation = np.corrcoef(self._yCal, y_cal_predict)[0][1]
        r2_cal = self._rf.score(self._xCal, self._yCal)
        rmse = mean_squared_error(self._yCal, y_cal_predict, squared=False)

        nsamples = self._xCal.shape[0]

        calibration_metrics = {'n_samples': nsamples, 'R': r_correlation, 'R2': r2_cal, 'RMSE': rmse}

        self.metrics['calibration'] = calibration_metrics  
    


    def cross_validate(self):
        
        r_correlation, r2_cv, rmse_cv, predicted_values = cross_validation(self._rf, self._xCal, self._yCal, self._cv, correlation_based=False)

        method = 'Leave One Out'
        if isinstance(self._cv, KFold):
            method = "{}-fold".format(self._cv.n_splits)
        
        cross_validation_metrics = {'R': r_correlation, 'R2': r2_cv, 'RMSE': rmse_cv, 'method': method, 'predicted_values': predicted_values }
        
        self.metrics['cross_validation'] = cross_validation_metrics
    
    def validate(self):

        r_correlation, r2_ve, rmse_ve, predicted_values = external_validation(self._rf, self._xVal, self._yVal, correlation_based=False)

        nsamples = self._xVal.shape[0]
        validation = {'R': r_correlation, 'R2': r2_ve, 'RMSE': rmse_ve, 'n_samples': nsamples, 'predicted_values': predicted_values}

        self.metrics['validation'] = validation

    def create_model(self):

        self.calibrate()
        self.cross_validate()
        self.validate()
    
    def save_results(self, path, name="results", out_table=False, plots=False):

        if path[-1] != '/':
            path += '/'
        
        if not os.path.exists(f"{path}{name}/"):
            os.mkdir(f"{path}{name}")

        with open(f"{path}{name}/model_information_{name}.txt", 'w') as out:
            out.write('==== Information of model ====\n\n')
            for parameter in self._rf.get_params():
                out.write(f"{parameter} = {self._rf.get_params()[parameter]}\n")
            out.write('\n')
            
            out.write('==== Calibration ====\n')
            out.write(f"n_samples = {self.metrics['calibration']['n_samples']}\n")
            out.write(f"Coefficiente of correlation (R) = {self.metrics['calibration']['R']:.5f}\n")
            out.write(f"Coefficient of determination (R2) = {self.metrics['calibration']['R2']:.5f}\n")
            out.write(f"Root mean squared error (RMSE) = {self.metrics['calibration']['RMSE']:.5f}\n\n")

            out.write('==== Cross-validation ====\n')
            try:
                out.write(f"Cross-validation type: {self.metrics['cross_validation']['method']}\n")
                out.write(f"Coefficiente of correlation (R) = {self.metrics['cross_validation']['R']:.5f}\n")
                out.write(f"Coefficient of determination (R2) = {self.metrics['cross_validation']['R2']:.5f}\n")
                out.write(f"Root mean squared error (RMSE) = {self.metrics['cross_validation']['RMSE']:.5f}\n\n")
            except:
                out.write('Cross-validation not performed.\n\n')
            
            out.write('==== Prediction ====\n')
            try:
                out.write(f"n_samples = {self.metrics['validation']['n_samples']}\n")
                out.write(f"Coefficiente of correlation (R) = {self.metrics['validation']['R']:.5f}\n")
                out.write(f"Coefficient of determination (R2) = {self.metrics['validation']['R2']:.5f}\n")
                out.write(f"Root mean squared error (RMSE) = {self.metrics['validation']['RMSE']:.5f}\n\n")
            except:
                out.write('Prediction not performed.\n\n')
        
        
        if plots == True:
            with PdfPages(f"{path}{name}/plots_{name}.pdf") as pdf:
                plt.rc('font', size=16)
                fig = plt.figure(figsize=(16, 12), dpi=100)
                gs = gridspec.GridSpec(2,2)
                
                ax1 = fig.add_subplot(gs[0,:2])
                ax1.plot(self._xCal.columns.astype('int'), self._rf.feature_importances_)
                ax1.set_ylabel('Importance')
                ax1.set_xlabel('Wavelength')
                ax1.set_title('Importance of variables')


                ax2 = fig.add_subplot(gs[1, 0])
                try:
                    ax2.scatter(self._yCal, self.metrics['cross_validation']['predicted_values'])
                    ax2.set_ylabel('Predicted')
                    ax2.set_xlabel('Reference')
                    ax2.set_title('Cross-validation')
                except:
                    ax2.plot([-1,1], c='black')
                    ax2.plot([1, -1], c='black')
                    ax2.axis('off')
                    ax2.set_title('Cross-validation not performed')
                
                ax3 = fig.add_subplot(gs[1, 1])
                try:
                    ax3.scatter(self._yVal, self.metrics['validation']['predicted_values'])
                    ax3.set_ylabel('Predicted')
                    ax3.set_xlabel('Reference')
                    ax3.set_title('Prediction')
                except:
                    ax3.plot([-1,1], c='black')
                    ax3.plot([1, -1], c='black')
                    ax3.axis('off')
                    ax3.set_title('Prediction not performed')
                
                plt.tight_layout(pad=1.5)
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
        
        if out_table == True:
            try:
                predictions = pd.DataFrame(np.vstack((self._yVal.values, self.metrics['validation']['predicted_values']))).T
                predictions.columns = ['Observed', 'Predicted']
                predictions.index = self._yVal.index

                predictions.to_csv(f"{path}{name}/predictions.csv", sep=';', decimal=',')
            except:
                pass

            try:
                cross_validation_prediction = pd.DataFrame(np.vstack((self._yCal.values, self.metrics['cross_validation']['predicted_values']))).T
                cross_validation_prediction.columns = ['Observed', 'Predicted']
                cross_validation_prediction.index = self._yCal.index

                cross_validation_prediction.to_csv(f"{path}{name}/predictions_CV.csv", sep=';', decimal=',')
            except:
                pass
                