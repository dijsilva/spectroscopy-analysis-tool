import pandas as pd

def make_average(dataset, number_of_repetions):
    n_samples = dataset.shape[0]

    X_average = pd.DataFrame()
    for index in list(range(0, n_samples, number_of_repetions)):
        avg_X = dataset.iloc[index: index + number_of_repetions,1:].mean(axis=0)
        avg_X = dataset.iloc[index:index+1,0].append(avg_X, ignore_index=True)
        X_average = pd.concat([X_average, avg_X], axis=1, ignore_index=True)


    
    X_average = X_average.transpose()
    X_average.index = list(range(0, n_samples, number_of_repetions))

    X_average.iloc[:,0] = X_average.iloc[:,0].astype('int')
    
    return X_average
        