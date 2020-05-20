__all__ = ['make_average', 'cross_validation', 'external_validation', 
           'classifier_cross_validation', 'classifier_external_validation', 
           'save_results_of_model']

from utils.average import make_average

# VALIDATIONS
from utils.model_validations import cross_validation
from utils.model_validations import external_validation
from utils.model_validations import classifier_cross_validation
from utils.model_validations import classifier_external_validation


# HANDLE RESULTS
from utils.handle_results import save_results_of_model