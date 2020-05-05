from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np

class PCA_LDA():
    def __init__(self, dataset, number_of_components, equal_probabilites=False, lda_random_state=123):
        self.dataset = dataset
        self.number_of_components  = number_of_components
        self.lda_random_state = lda_random_state
        self.equal_probabilites = equal_probabilites

        if not isinstance(self.dataset, pd.DataFrame):
            raise ValueError('The dataset should be a pd.DataFrame.')

        if type(number_of_components) not in [int]:
            raise ValueError('number_of_components should be a positive integer.')

        self.x = dataset.iloc[:, 2:]
        self.y = dataset.iloc[:, 1]

        self.metrics = {}
    
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
        cm = confusion_matrix(self.y, self.predictions)
        cm = pd.DataFrame(cm)

        index_columns = self.y.value_counts().sort_index(axis=0).index
        cm.index = index_columns
        cm.columns = index_columns

        calibration_metrics = {'accuracy': accuracy, 'confusion_matrix': cm.T, 'n_components': self.number_of_components, 'priors': self.priors}

        self.metrics['calibration'] = calibration_metrics
    

    def create_model(self):
        
        self.pca()
        self.lda()