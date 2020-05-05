from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PCA_LDA():
    def __init__(self, dataset, number_of_components, equal_probabilites=False, lda_random_state=123):
        self.dataset = dataset
        self.number_of_components  = number_of_components
        self.lda_random_state = lda_random_state
        self.equal_probabilites = equal_probabilites
        self.metrics = {}

        if not isinstance(self.dataset, pd.DataFrame):
            raise ValueError('The dataset should be a pd.DataFrame.')

        if type(number_of_components) not in [int]:
            raise ValueError('number_of_components should be a positive integer.')

        if self.equal_probabilites not in [True, False]:
            raise ValueError('equal_probabilites should be a boolean value')

        if (type(self.lda_random_state) not in [int]):
            raise ValueError('lda_random_state should be a integer')

        self.x = dataset.iloc[:, 2:]
        self.y = dataset.iloc[:, 1]
    
    def pca(self):
        self._pca = PCA(n_components=self.number_of_components, tol=0.0000000001, random_state=self.lda_random_state, svd_solver="full")
        
        self._xReduced = self._pca.fit_transform(self.x)
    
    def lda(self):
        
        if self.equal_probabilites == True:
            n_class = len(self.y.unique())
            self.priors = np.full((n_class, ),  1 / n_class)
        else:
            self.priors = self.y.value_counts(normalize=True)
            self.priors = np.array(self.priors.sort_index(axis=0))

        self._lda = LinearDiscriminantAnalysis(n_components=self.number_of_components, tol=1e-8, priors=self.priors)

        self._lda.fit(self._xReduced, self.y)

        self.predictions = self._lda.predict(self._xReduced)

        accuracy = accuracy_score(self.y, self.predictions)
        self.cm = confusion_matrix(self.y, self.predictions)
        self.cm = pd.DataFrame(self.cm)

        index_columns = self.y.value_counts().sort_index(axis=0).index
        self.cm.index = index_columns
        self.cm.columns = index_columns

        calibration_metrics = {'accuracy': accuracy, 'confusion_matrix': self.cm, 'n_components': self.number_of_components, 'priors': self.priors}

        self.metrics['calibration'] = calibration_metrics
    
    def prediction(self):

    
    def plot_confusion_matrix(self, title='Confusion Matrix', cbar=True):
        sn.heatmap(self.cm, annot=True, cmap='Greys', linewidths=0.7, linecolor='black', cbar=cbar, square=True)
        plt.title(title, pad=20.0)
        plt.ylabel('Reference')
        plt.xlabel('Predicted')
        plt.tight_layout(pad=1.0)
        return plt

    def create_model(self):
        
        self.pca()
        self.lda()