import pandas as pd
from scipy.signal import savgol_filter

def snv(dataset):
    rows = dataset.shape[0]

    for row in range(rows):
        mean = dataset.iloc[row, :].mean(axis=0)
        std = dataset.iloc[row, :].std(axis=0)

        dataset.iloc[row, :] = (dataset.iloc[row, :] - mean) / std
    
    return dataset


def sg(dataset, differentiation, window_size, polynominal_order):
    
    sg_df = savgol_filter(dataset, window_length=window_size, polyorder=polynominal_order, deriv=differentiation, axis=-1)

    df = pd.DataFrame(sg_df)

    df.columns = dataset.columns
    df.index = dataset.index

    gap = window_size // 2
    return df.iloc[:,gap:-gap]

def snv_sg(dataset, differentiation, window_size, polynominal_order, ):
    
    snv_dataset = snv(dataset)
    snv_sg_df = savgol_filter(snv_dataset, window_length=window_size, polyorder=polynominal_order, deriv=differentiation, axis=-1)

    df = pd.DataFrame(snv_sg_df)

    df.columns = dataset.columns
    df.index = dataset.index

    gap = window_size // 2
    return df.iloc[:,gap:-gap]

    return df
