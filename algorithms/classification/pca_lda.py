from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, LeaveOneOut
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import classifier_cross_validation, classifier_external_validation

class PCA_LDA():
    def __init__(self, dataset, number_of_components, cross_validation_type='loo', split_for_validation=None, data_validation=None, equal_probabilites=False, lda_random_state=123):
        self.dataset = dataset
        self.number_of_components  = number_of_components
        self.split_for_validation = split_for_validation
        self.data_validation = data_validation
        self.equal_probabilites = equal_probabilites
        self.lda_random_state = lda_random_state
        self.metrics = {}
        self._cv = None

        if not isinstance(self.dataset, pd.DataFrame):
            raise ValueError('The dataset should be a pd.DataFrame.')

        if (self.data_validation is None) and (self.split_for_validation is None):
            raise ValueError('Should be defined the samples for validation or size of test size for split the dataset.')

        if type(number_of_components) not in [int]:
            raise ValueError('number_of_components should be a positive integer.')

        if (not self.split_for_validation is None) and (self.data_validation is None):
            if self.split_for_validation == 'all':
                self._xCal = self.dataset.iloc[:, 2:]
                self._yCal = self.dataset.iloc[:, 1]
            elif isinstance(self.split_for_validation, float):
                self._xCal, self._xVal, self._yCal, self._yVal = train_test_split(self.dataset.iloc[:, 2:], self.dataset.iloc[:, 1], test_size=split_for_validation, random_state=lda_random_state)
            else:
                raise ValueError("split_for_validation need be a float value between 0 and 1 for split dataset. Use 1 for calibrate with all samples of dataset.")

        if not self.data_validation is None:
            if isinstance(self.data_validation, pd.DataFrame):
                self._xCal = self.dataset.iloc[:, 2:]
                self._yCal = self.dataset.iloc[:, 1]
                self._xVal = self.data_validation.iloc[:, 2:]
                self._yVal = self.data_validation.iloc[:, 1]
            else:
                raise ValueError("data_validation need be a pandas dataframe")

        if self.equal_probabilites not in [True, False]:
            raise ValueError('equal_probabilites should be a boolean value')

        if (type(self.lda_random_state) not in [int]):
            raise ValueError('lda_random_state should be a integer')

        if isinstance(cross_validation_type, str):
            if cross_validation_type == "loo":
                self._cv = LeaveOneOut()
            elif (type(cross_validation_type) in [int]) and (cross_validation_type > 0):
                self._cv = cross_validation_type
            else:
                raise ValueError("The cross_validation_type should be a positive integer for k-fold method ou 'loo' for leave one out cross validation.")
    
    
    def calibrate(self):
        self._pca = PCA(n_components=self.number_of_components, tol=0.0000000001, random_state=self.lda_random_state, svd_solver="full")
        
        self._xReduced = self._pca.fit_transform(self._xCal)
        
        if self.equal_probabilites == True:
            n_class = len(self._yCal.unique())
            self.priors = np.full((n_class, ),  1 / n_class)
        else:
            self.priors = self._yCal.value_counts(normalize=True)
            self.priors = np.array(self.priors.sort_index(axis=0))

        self._lda = LinearDiscriminantAnalysis(n_components=self.number_of_components, tol=1e-8, priors=self.priors)

        self._lda.fit(self._xReduced, self._yCal)

        self.predictions = self._lda.predict(self._xReduced)

        accuracy = accuracy_score(self._yCal, self.predictions)
        cm = confusion_matrix(self._yCal, self.predictions)
        cm = pd.DataFrame(cm)

        index_columns = self._yCal.value_counts().sort_index(axis=0).index
        cm.index = index_columns
        cm.columns = index_columns

        n_samples = self._yCal.shape[0]

        calibration_metrics = {'accuracy': accuracy, 'confusion_matrix': cm, 'n_samples': n_samples, 'n_components': self.number_of_components, 'priors': self.priors}

        self.metrics['calibration'] = calibration_metrics
    
    def cross_validate(self):
        
        accuracy, cm, predicted_values = classifier_cross_validation(self._lda, self._xReduced, self._yCal, cv=self._cv)

        method = 'LOO'
        if isinstance(self._cv, int):
            method = "{}-fold".format(self._cv)

        cross_validation_metrics = {'accuracy': accuracy, 'confusion_matrix': cm, 'method': method, 'predicted_values': predicted_values}

        self.metrics['cross_validation'] = cross_validation_metrics
    

    def validate(self):

        self._pca_val = PCA(n_components=self.number_of_components, tol=0.0000000001, random_state=self.lda_random_state, svd_solver="full")

        self.xValReduced = self._pca_val.fit_transform(self._xVal)

        accuracy, cm, predicted_values = classifier_external_validation(self._lda, self.xValReduced, self._yVal)

        nsamples = self._xVal.shape[0]
        validation = {'accuracy': accuracy, 'confusion_matrix': cm, 'n_samples': nsamples, 'predicted_values': predicted_values}

        self.metrics['validation'] = validation
    
    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cbar=True):
        sn.heatmap(cm, annot=True, cmap='Greys', linewidths=0.7, linecolor='black', cbar=cbar, square=True, fmt='g')
        plt.title(title, pad=20.0)
        plt.ylabel('Reference')
        plt.xlabel('Predicted')
        plt.tight_layout(pad=1.0)
        return plt
        

    def create_model(self):
        
        self.calibrate()
        self.cross_validate()
        self.validate()