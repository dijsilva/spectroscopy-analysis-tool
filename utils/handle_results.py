def save_results_of_model(model, path, name="results", out_table=False, plots=False):
    if path[-1] != '/':
            path += '/'
        
        if not os.path.exists(f"{path}{name}/"):
            os.mkdir(f"{path}{name}")

        with open(f"{path}{name}/model_information_{name}.txt", 'w') as out:
            out.write('==== Information of model ====\n\n')
            for parameter in model._rf.get_params():
                out.write(f"{parameter} = {model._rf.get_params()[parameter]}\n")
            out.write('\n')
            
            out.write('==== Calibration ====\n')
            out.write(f"n_samples = {model.metrics['calibration']['n_samples']}\n")
            out.write(f"Coefficient  of correlation (R) = {model.metrics['calibration']['R']:.5f}\n")
            out.write(f"Coefficient of determination (R2) = {model.metrics['calibration']['R2']:.5f}\n")
            out.write(f"Root mean squared error (RMSE) = {model.metrics['calibration']['RMSE']:.5f}\n\n")

            out.write('==== Cross-validation ====\n')
            try:
                out.write(f"Cross-validation type: {model.metrics['cross_validation']['method']}\n")
                out.write(f"Coefficient of correlation (R) = {model.metrics['cross_validation']['R']:.5f}\n")
                out.write(f"Coefficient of determination (R2) = {model.metrics['cross_validation']['R2']:.5f}\n")
                out.write(f"Root mean squared error (RMSE) = {model.metrics['cross_validation']['RMSE']:.5f}\n\n")
            except:
                out.write('Cross-validation not performed.\n\n')
            
            out.write('==== Prediction ====\n')
            try:
                out.write(f"n_samples = {model.metrics['validation']['n_samples']}\n")
                out.write(f"Coefficient of correlation (R) = {model.metrics['validation']['R']:.5f}\n")
                out.write(f"Coefficient of determination (R2) = {model.metrics['validation']['R2']:.5f}\n")
                out.write(f"Root mean squared error (RMSE) = {model.metrics['validation']['RMSE']:.5f}\n\n")
            except:
                out.write('Prediction not performed.\n\n')
        
        
        if plots == True:
            with PdfPages(f"{path}{name}/plots_{name}.pdf") as pdf:
                plt.rc('font', size=16)
                fig = plt.figure(figsize=(16, 12), dpi=100)
                gs = gridspec.GridSpec(2,2)
                
                ax1 = fig.add_subplot(gs[0,:2])
                ax1.plot(model._xCal.columns.astype('int'), model._rf.feature_importances_)
                ax1.set_ylabel('Importance')
                ax1.set_xlabel('Wavelength')
                ax1.set_title('Importance of variables')


                ax2 = fig.add_subplot(gs[1, 0])
                try:
                    ax2.scatter(model._yCal, model.metrics['cross_validation']['predicted_values'])
                    ax2.set_ylabel('Predicted')
                    ax2.set_xlabel('Reference')
                    ax2.set_title('Cross-validation')
                except:
                    ax2.plot([-1,1], c='black')
                    ax2.plot([1, -1], c='black')
                    ax2.axis('off')
                    ax2.set_title('Cross-validation not performed')
                
                ax3 = fig.add_subplot(gs[1, 1])
                try:
                    ax3.scatter(model._yVal, model.metrics['validation']['predicted_values'])
                    ax3.set_ylabel('Predicted')
                    ax3.set_xlabel('Reference')
                    ax3.set_title('Prediction')
                except:
                    ax3.plot([-1,1], c='black')
                    ax3.plot([1, -1], c='black')
                    ax3.axis('off')
                    ax3.set_title('Prediction not performed')
                
                plt.tight_layout(pad=1.5)
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
        
        if out_table == True:
            try:
                predictions = pd.DataFrame(np.vstack((model._yVal.values, model.metrics['validation']['predicted_values']))).T
                predictions.columns = ['Observed', 'Predicted']
                predictions.index = model._yVal.index

                predictions.to_csv(f"{path}{name}/predictions.csv", sep=';', decimal=',')
            except:
                pass

            try:
                cross_validation_prediction = pd.DataFrame(np.vstack((model._yCal.values, model.metrics['cross_validation']['predicted_values']))).T
                cross_validation_prediction.columns = ['Observed', 'Predicted']
                cross_validation_prediction.index = model._yCal.index

                cross_validation_prediction.to_csv(f"{path}{name}/predictions_CV.csv", sep=';', decimal=',')
            except:
                pass
                