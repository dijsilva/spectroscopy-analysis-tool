import pandas as pd
df = pd.read_csv('/home/dsilva/foss_para_arff_calibracao.csv', delimiter=';', decimal=',')

x = df.iloc[:, 2:]
y = df.iloc[:, 1]
def make_average(x, y, number_of_repetions):
    n_samples = x.shape[0]

    X_average = pd.DataFrame()
    for index in list(range(0, n_samples, number_of_repetions)):
        avg_X = x.iloc[index: index + number_of_repetions,:].mean(axis=0)
        X_average = pd.concat([X_average, avg_X], axis=1)


    
    X_average = X_average.transpose()
    X_average.index = list(range(0, n_samples, number_of_repetions))
    
    return X_average
        