import pandas as pd
import numpy as np

def make_average(dataset, number_of_repetions, start_spectra):

    if not isinstance(dataset, pd.DataFrame):
        raise ValueError('The dataset should be a dataframe.')

    if type(number_of_repetions) not in [int]:
        raise ValueError('The number_of_repetions should be integer.')

    if type(start_spectra) not in [int]:
        raise ValueError('The start_spectra should be integer that references a column which spectra starts.')

    if number_of_repetions <= 0:
        raise ValueError('The number_of_repetions cannot be negative.')
    
    n_samples = dataset.shape[0]

    #X_average = pd.DataFrame()
    X_average = pd.DataFrame(np.zeros((int(n_samples / number_of_repetions), dataset.shape[1])))

    cont = 0
    for index in list(range(0, n_samples, number_of_repetions)):
        X_average.iloc[cont, start_spectra :] = dataset.iloc[index : index + number_of_repetions, start_spectra :].mean(axis=0).values
        X_average.iloc[cont, : start_spectra] = dataset.iloc[index, : start_spectra].values
        cont += 1

    X_average.index = list(range(0, n_samples, number_of_repetions))
    X_average.columns = dataset.columns
    X_average.iloc[:,0] = X_average.iloc[:,0].astype('int')
    
    return X_average
        