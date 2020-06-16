# The project is in progress...

## A graphical user interface for building mathematical models with chemometrics data
A set of algorithms for the purpose of performing sample predictions and classifications using spectroscopy data

### Algorithms that can already be used: 
#### Regression
- Partial least square regression (PLS)
- Support Vector Machine regression (SVR)
- Principal Component Regression (PCR)
- Random Forest.

#### Classification
- Principal component analysis + linear discrminant analysis (PCA-LDA)

### Some details
When the algorithms in this tool have the `search_hypermatameters` method, that method can be used to search for the best set of hyperparameters based on the tool [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) of sklearn library.

In the run.py file, should be defined some variables like 
- `FOLDER_BASE` = This is a string that define the directory where the output will be saved.
- `ANALYSIS` = This is a string that define the name of folder created for save output.
- `save_results` = If the tool should be save the results or not (is a bool variable)
- `MAKE_AVERAGE` = If necessary, samples are averaged. if `True`, it must be informed in which column the variables begin and the number of samples must be used to perform an average.  (is a bool variable)