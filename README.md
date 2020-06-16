# A graphical user interface for building mathematical models with chemometrics data

A set of algorithms for the purpose of performing sample predictions and classifications using spectroscopy data

## The project is in progress...

## Algorithms that can already be used: 

### Regression
- Partial least squares regression (PLS)
- Support Vector Machine regression (SVR)
- Principal Component Regression (PCR)
- Random Forest.

### Classification
- Principal component analysis + linear discrminant analysis (PCA-LDA)

## Some details
When the algorithms in this tool have the search_hypermatameters method, that method can be used to search for the best set of hyperparameters based on the tool [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) of sklearn library.