import pandas as pd

def snv(dataset):
    rows = dataset.shape[0]

    for row in range(rows):
        mean = dataset.iloc[row, :].mean(axis=0)
        std = dataset.iloc[row, :].std(axis=0)

        dataset.iloc[row, :] = (dataset.iloc[row, :] - mean) / std
    
    return dataset
