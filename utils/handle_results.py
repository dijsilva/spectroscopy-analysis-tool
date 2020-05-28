import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


def save_results_of_model(model_instance, path, name="results", out_table=False, plots=False, out_performance=False, coefficients_of_model='plsr'):
    if path[-1] != '/':
            path += '/'
        
    if not os.path.exists(f"{path}{name}/"):
        os.mkdir(f"{path}{name}")

    with open(f"{path}{name}/model_information_{name}.txt", 'w') as out:
        out.write('==== Information of model ====\n\n')
        for parameter in model_instance.model.get_params():
            out.write(f"{parameter} = {model_instance.model.get_params()[parameter]}\n")
        out.write('\n')
        
        out.write('==== Calibration ====\n')
        out.write(f"n_samples = {model_instance.metrics['calibration']['n_samples']}\n")
        out.write(f"Coefficient  of correlation (R) = {model_instance.metrics['calibration']['R']:.5f}\n")
        out.write(f"Coefficient of determination (R2) = {model_instance.metrics['calibration']['R2']:.5f}\n")
        out.write(f"Root mean squared error (RMSE) = {model_instance.metrics['calibration']['RMSE']:.5f}\n\n")

        out.write('==== Cross-validation ====\n')
        try:
            out.write(f"Cross-validation type: {model_instance.metrics['cross_validation']['method']}\n")
            out.write(f"Coefficient of correlation (R) = {model_instance.metrics['cross_validation']['R']:.5f}\n")
            out.write(f"Coefficient of determination (R2) = {model_instance.metrics['cross_validation']['R2']:.5f}\n")
            out.write(f"Root mean squared error (RMSE) = {model_instance.metrics['cross_validation']['RMSE']:.5f}\n\n")
        except:
            out.write('Cross-validation not performed.\n\n')
        
        out.write('==== Prediction ====\n')
        try:
            out.write(f"n_samples = {model_instance.metrics['validation']['n_samples']}\n")
            out.write(f"Coefficient of correlation (R) = {model_instance.metrics['validation']['R']:.5f}\n")
            out.write(f"Coefficient of determination (R2) = {model_instance.metrics['validation']['R2']:.5f}\n")
            out.write(f"Root mean squared error (RMSE) = {model_instance.metrics['validation']['RMSE']:.5f}\n\n")
        except:
            out.write('Prediction not performed.\n\n')
    
    
    if plots == True:
        with PdfPages(f"{path}{name}/plots_{name}.pdf") as pdf:
            
            fig1 = plt.figure(figsize=(16, 12), dpi=100)
            plt.rc('font', size=16)
            plt.tight_layout(pad=0.5)
            gs = gridspec.GridSpec(2,2)
            
            ax1 = fig1.add_subplot(gs[0,:2])
            if coefficients_of_model == 'random_forest':
                ax1.plot(model_instance._xCal.columns.astype('int'), model_instance.model.feature_importances_)
                ax1.set_ylabel('Importance')
                ax1.set_xlabel('Wavelength')
                ax1.set_title('Importance of variables')
            elif coefficients_of_model == 'plsr':
                ax1.plot(model_instance._xCal.columns.astype('int'), model_instance.model.coef_)
                x_for_line = [0] * model_instance.model.coef_.shape[0]
                ax1.plot(model_instance._xCal.columns.astype('int'), x_for_line, c='black')
                ax1.set_ylabel('Value of coefficient')
                ax1.set_xlabel('Wavelength')
                ax1.set_title('Coefficients')
            elif coefficients_of_model == 'svr' and model_instance.model.kernel == 'linear':
                ax1.plot(model_instance._xCal.columns.astype('int'), model_instance.model.coef_[0])
                x_for_line = [0] * model_instance.model.coef_[0].shape[0]
                ax1.plot(model_instance._xCal.columns.astype('int'), x_for_line, c='black')
                ax1.set_ylabel('Value of coefficient')
                ax1.set_xlabel('Wavelength')
                ax1.set_title('Coefficients')
            elif coefficients_of_model == 'svr' and model_instance.model.kernel != 'linear':
                ax1.plot([1, -1], c='black')
                ax1.plot([-1, 1], c='black')
                ax1.axis('off')
                ax1.set_ylabel('Value of coefficient')
                ax1.set_xlabel('Wavelength')
                ax1.set_title('Coefficients')

            ax2 = fig1.add_subplot(gs[1, 0])
            try:
                ax2.scatter(model_instance._yCal, model_instance.metrics['cross_validation']['predicted_values'])
                ax2.set_ylabel('Predicted')
                ax2.set_xlabel('Reference')
                ax2.set_title('Cross-validation')
            except:
                ax2.plot([-1,1], c='black')
                ax2.plot([1, -1], c='black')
                ax2.axis('off')
                ax2.set_title('Cross-validation not performed')
            
            ax3 = fig1.add_subplot(gs[1, 1])
            try:
                ax3.scatter(model_instance._yVal, model_instance.metrics['validation']['predicted_values'])
                ax3.set_ylabel('Predicted')
                ax3.set_xlabel('Reference')
                ax3.set_title('Prediction')
            except:
                ax3.plot([-1,1], c='black')
                ax3.plot([1, -1], c='black')
                ax3.axis('off')
                ax3.set_title('Prediction not performed')
            
            plt.tight_layout(pad=2.0)  # saves the current figure into a pdf page
            pdf.savefig(fig1)  # saves the current figure into a pdf page
            plt.close()

            fig2 = plt.figure(figsize=(16, 12), dpi=100)
            plt.rc('font', size=16)
            gs = gridspec.GridSpec(2,2)
            ax4 = fig2.add_subplot(gs[0, :])
            try:
                ax4.plot(model_instance._xCal.columns.astype('int'), model_instance._xCal.T.values)
                ax4.set_ylabel('Absorbance')
                ax4.set_xlabel('Wavelength')
                ax4.set_title('Spectra of calibration data')
            except:
                ax4.plot([-1,1], c='black')
                ax4.plot([1, -1], c='black')
                ax4.axis('off')
                ax4.set_title('Plot not performed')
            
            ax5 = fig2.add_subplot(gs[1, :])
            try:
                ax5.plot(model_instance._xVal.columns.astype('int'), model_instance._xVal.T.values)
                ax5.set_ylabel('Absorbance')
                ax5.set_xlabel('Wavelength')
                ax5.set_title('Spectra of prediction data')
            except:
                ax5.plot([-1,1], c='black')
                ax5.plot([1, -1], c='black')
                ax5.axis('off')
                ax5.set_title('Plot not performed')
            
            plt.tight_layout(pad=2.0)  # saves the current figure into a pdf page
            pdf.savefig(fig2)
            plt.close()





            if out_performance == True:

                fig3 = plt.figure(figsize=(16, 12), dpi=100)
                plt.rc('font', size=16)
                gs = gridspec.GridSpec(2,2)

                ax6 = fig3.add_subplot(gs[0, 0])
                try:
                    ax6.plot(model_instance._perfomance['components'], model_instance._perfomance['cross_validation']['RMSE'])
                    ax6.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax6.set_ylabel('RMSE')
                    ax6.set_xlabel('Components')
                    ax6.set_title('RMSE of Cross-validation')
                except:
                    ax6.plot([-1,1], c='black')
                    ax6.plot([1, -1], c='black')
                    ax6.axis('off')
                    ax6.set_title('performance of cv not analyzed')
                

                ax7 = fig3.add_subplot(gs[0, 1])
                try:
                    ax7.plot(model_instance._perfomance['components'], model_instance._perfomance['cross_validation']['R2'])
                    ax7.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax7.set_ylabel('R^2')
                    ax7.set_xlabel('Components')
                    ax7.set_title('R^2 of Cross-validation')
                except:
                    ax7.plot([-1,1], c='black')
                    ax7.plot([1, -1], c='black')
                    ax7.axis('off')
                    ax7.set_title('performance of cv not analyzed')
                

                ax8 = fig3.add_subplot(gs[1, 0])
                try:
                    ax8.plot(model_instance._perfomance['components'], model_instance._perfomance['validation']['RMSE'])
                    ax8.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax8.set_ylabel('RMSE')
                    ax8.set_xlabel('Components')
                    ax8.set_title('RMSE of prediction')
                except:
                    ax8.plot([-1,1], c='black')
                    ax8.plot([1, -1], c='black')
                    ax8.axis('off')
                    ax8.set_title('performance of prediction not analyzed')
                

                ax9 = fig3.add_subplot(gs[1, 1])
                try:
                    ax9.plot(model_instance._perfomance['components'], model_instance._perfomance['validation']['R2'])
                    ax9.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax9.set_ylabel('R^2')
                    ax9.set_xlabel('Components')
                    ax9.set_title('R^2 of prediction')
                except:
                    ax9.plot([-1,1], c='black')
                    ax9.plot([1, -1], c='black')
                    ax9.axis('off')
                    ax9.set_title('performance of prediction not analyzed')
                
                plt.tight_layout(pad=2.0)  # saves the current figure into a pdf page
                pdf.savefig(fig3)
                plt.close()

    
    if out_table == True:
        try:
            predictions = pd.DataFrame(np.vstack((model_instance._yVal.values, model_instance.metrics['validation']['predicted_values']))).T
            predictions.columns = ['Observed', 'Predicted']
            predictions.index = model_instance._yVal.index

            predictions.to_csv(f"{path}{name}/predictions.csv", sep=';', decimal=',')
        except:
            pass

        try:
            cross_validation_prediction = pd.DataFrame(np.vstack((model_instance._yCal.values, model_instance.metrics['cross_validation']['predicted_values']))).T
            cross_validation_prediction.columns = ['Observed', 'Predicted']
            cross_validation_prediction.index = model_instance._yCal.index

            cross_validation_prediction.to_csv(f"{path}{name}/predictions_CV.csv", sep=';', decimal=',')
        except:
            pass

            