def experimentBaselines(baseline_models):
    from ensemble import DeepEnsembleTest,makeEnsemblePrediction
    from sklearn.metrics import accuracy_score
    from testFunctions import testCalibrationBins, testCalibrationError
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # get model predictions
    ensemble_models_probabilities, ensemble_models_ensemble_predictions, all_labels = DeepEnsembleTest(baseline_models,
                                                                                                       softmax=False)
    ensemble_models_probabilities_softmax, ensemble_models_predictions_softmax, all_labels_softmax = DeepEnsembleTest(
        baseline_models, softmax=True)

    model_accuracies = []
    calibration_args = []
    calibration_errors = []

    # get models individual accuracy and compute calibration errors
    for model_number in range(len(baseline_models)):
        # accuracies
        model_accuracy = accuracy_score(all_labels, ensemble_models_ensemble_predictions[model_number])
        error = 1.96 * np.sqrt((model_accuracy * (1 - model_accuracy)) / len(all_labels))
        model_accuracies += [(model_accuracy, error)]

        # calibration errors
        calibration_args += [testCalibrationBins(ensemble_models_probabilities_softmax[model_number],
                                                 ensemble_models_predictions_softmax[model_number], all_labels_softmax)]
        calibration_errors += [testCalibrationError(*calibration_args[model_number])]

    ensemble_accuracies = []
    ensemble_calibration_args = []
    ensemble_calibration_errors = []

    # get ensemble accuracy and compute calibration errors
    for ensemble_size in range(1, len(baseline_models) + 1):
        # accuracies
        ensemble_accuracy = accuracy_score(all_labels, makeEnsemblePrediction(
            ensemble_models_probabilities_softmax[:ensemble_size])[2])
        ensemble_error = 1.96 * np.sqrt((ensemble_accuracy * (1 - ensemble_accuracy)) / len(all_labels_softmax))
        ensemble_accuracies += [(ensemble_accuracy, ensemble_error)]

        # calibration errors
        ensemble_calibration_args += [
            testCalibrationBins(makeEnsemblePrediction(ensemble_models_probabilities_softmax[:ensemble_size])[0],
                                makeEnsemblePrediction(ensemble_models_probabilities_softmax[:ensemble_size])[2],
                                all_labels_softmax)]
        ensemble_calibration_errors += [testCalibrationError(*ensemble_calibration_args[ensemble_size - 1])]

    torch.save(model_accuracies, 'baseline_model_accuracies')
    torch.save(calibration_args, 'baseline_calibration_args')
    torch.save(calibration_errors, 'baseline_model_errors')

    torch.save(ensemble_accuracies, 'ensemble_model_accuracies')
    torch.save(ensemble_calibration_args, 'ensemble_calibration_args')
    torch.save(ensemble_calibration_errors, 'ensemble_model_errors')

    plt.axhline(y=0, xmin=0.0, color='black', linestyle='dotted', linewidth=3)
    [plt.plot(ensemble_calibration_args[i][0], np.array(calibration_args[i][2]) - np.array(calibration_args[i][1])) for
     i in range(len(calibration_args))]
    # plt.legend(['M=%i'% i if i>0 else 'Perfect Calibration' for i in range(0,len(ensemble_calibration_args)+1)],loc="lower center",bbox_to_anchor=(0.5, -0.45),fancybox=True, shadow=True, ncol=4)
    plt.grid()
    plt.xlabel('Confidence (max prob)')
    plt.ylabel('Confidence - Accuracy')
    plt.yticks(np.arange(-0.4, 0.5, step=0.1))
    plt.title('Reliability diagram: Baselines', size=15, pad=10)
    plt.savefig('Reliability_diagram_baselines', bbox_inches='tight', dpi=600)
    plt.close()

    plt.axhline(y=0, xmin=0.0, color='black', linestyle='dotted', linewidth=4)
    [plt.plot(ensemble_calibration_args[i][0],
              np.array(ensemble_calibration_args[i][2]) - np.array(ensemble_calibration_args[i][1])) for i in
     range(len(ensemble_calibration_args))]
    # plt.legend(['M=%i'% i if i>0 else 'Perfect Calibration' for i in range(0,len(ensemble_calibration_args)+1)],loc="lower center",bbox_to_anchor=(0.5, -0.45),fancybox=True, shadow=True, ncol=4)
    plt.grid()
    plt.xlabel('Confidence (max prob)')
    plt.ylabel('Confidence - Accuracy')
    plt.title('Reliability diagram: Ensembles', size=15, pad=10)
    plt.yticks(np.arange(-0.4, 0.5, step=0.1))
    plt.savefig('Reliability_diagram_ensembles', bbox_inches='tight', dpi=600)
    plt.close()

    plt.grid()
    plt.plot(np.arange(1, 11), [ensemble_accuracies[i][0] for i in range(len(ensemble_accuracies))], color='red')
    plt.axhline(y=np.mean([model_accuracies[i][0] for i in range(len(model_accuracies))]), xmin=0, color='grey',
                ls='dashed', lw=3)
    plt.scatter(np.arange(1, 11), [model_accuracies[i][0] for i in range(len(model_accuracies))], color='black',
                ls='--')
    plt.scatter(np.arange(1, 11), [ensemble_accuracies[i][0] for i in range(len(ensemble_accuracies))], marker='s',
                color='red')
    plt.title('Ensemble classification accuracy \n wrt. ensemble size (M)', size=16)
    plt.legend(['Ensemble', 'Average model accuracy', 'Individual baseline acc.'], bbox_to_anchor=(1.15, -0.15),
               fancybox=True, shadow=True, ncol=3)
    plt.xlabel('Ensemble size (M)')
    plt.ylabel('Acc.')
    plt.yticks(np.arange(0.80, 0.90, 0.01))
    plt.savefig('Ensemble_acc', bbox_inches='tight', dpi=600)
    plt.close()

    plt.grid()
    plt.plot(np.arange(1, 11), ensemble_calibration_errors, linewidth=2)
    plt.scatter(np.arange(1, 11), [calibration_errors[i][0] for i in range(len(model_accuracies))], color='sienna',
                ls='--')
    plt.scatter(np.arange(1, 11), [calibration_errors[i][1] for i in range(len(model_accuracies))], color='grey',
                ls='--')
    plt.scatter(np.arange(1, 11), [ensemble_calibration_errors[i][0] for i in range(len(ensemble_calibration_errors))],
                marker='s')
    plt.scatter(np.arange(1, 11), [ensemble_calibration_errors[i][1] for i in range(len(ensemble_calibration_errors))],
                marker='s')
    plt.axhline(y=np.mean([calibration_errors[i][0] for i in range(len(model_accuracies))]), xmin=0, color='sienna',
                ls='dashed', lw=2)
    plt.axhline(y=np.mean([calibration_errors[i][1] for i in range(len(model_accuracies))]), xmin=0, color='grey',
                ls='dashed', lw=2)
    plt.title('Ensemble calibration wrt. ensemble size (M)', size=16)
    plt.legend(['Ensemble ECE','Ensemble MCE','Average model ECE','Moving average model MCE','Individual model ECE','Individual model MCE'],bbox_to_anchor=(1.15, -0.15),fancybox=True, shadow=True, ncol=3)
    plt.xlabel('Ensemble size (M)')
    plt.ylabel('Calibration error')
    plt.savefig('Ensemble_calibration', bbox_inches='tight', dpi=600)
    plt.close()
def experimentSWA(SWA_models):
    from ensemble import DeepEnsembleTest, makeEnsemblePrediction
    from sklearn.metrics import accuracy_score
    from testFunctions import testCalibrationBins, testCalibrationError
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # get model predictions
    ensemble_models_probabilities, ensemble_models_ensemble_predictions, all_labels = DeepEnsembleTest(SWA_models,
                                                                                                       softmax=False)
    ensemble_models_probabilities_softmax, ensemble_models_predictions_softmax, all_labels_softmax = DeepEnsembleTest(
        SWA_models, softmax=True)

    model_accuracies = []
    calibration_args = []
    calibration_errors = []

    # get models individual accuracy and compute calibration errors
    for model_number in range(len(SWA_models)):
        # accuracies
        model_accuracy = accuracy_score(all_labels, ensemble_models_ensemble_predictions[model_number])
        error = 1.96 * np.sqrt((model_accuracy * (1 - model_accuracy)) / len(all_labels))
        model_accuracies += [(model_accuracy, error)]

        # calibration errors
        calibration_args += [testCalibrationBins(ensemble_models_probabilities_softmax[model_number],
                                                 ensemble_models_predictions_softmax[model_number], all_labels_softmax)]
        calibration_errors += [testCalibrationError(*calibration_args[model_number])]

    ensemble_accuracies = []
    ensemble_calibration_args = []
    ensemble_calibration_errors = []

    # get ensemble accuracy and compute calibration errors
    for ensemble_size in range(1, len(SWA_models) + 1):
        # accuracies
        ensemble_accuracy = accuracy_score(all_labels, makeEnsemblePrediction(
            ensemble_models_probabilities_softmax[:ensemble_size])[2])
        ensemble_error = 1.96 * np.sqrt((ensemble_accuracy * (1 - ensemble_accuracy)) / len(all_labels_softmax))
        ensemble_accuracies += [(ensemble_accuracy, ensemble_error)]

        # calibration errors
        ensemble_calibration_args += [
            testCalibrationBins(makeEnsemblePrediction(ensemble_models_probabilities_softmax[:ensemble_size])[0],
                                makeEnsemblePrediction(ensemble_models_probabilities_softmax[:ensemble_size])[2],
                                all_labels_softmax)]
        ensemble_calibration_errors += [testCalibrationError(*ensemble_calibration_args[ensemble_size - 1])]

    torch.save(model_accuracies, 'SWA_model_accuracies')
    torch.save(calibration_args, 'SWA_calibration_args')
    torch.save(calibration_errors, 'SWA_model_errors')

    torch.save(ensemble_accuracies, 'Multi_SWA_model_accuracies')
    torch.save(ensemble_calibration_args, 'Multi_SWA_calibration_args')
    torch.save(ensemble_calibration_errors, 'Multi_SWA_model_errors')

    plt.axhline(y=0, xmin=0.0, color='black', linestyle='dotted', linewidth=3)
    [plt.plot(ensemble_calibration_args[i][0], np.array(calibration_args[i][2]) - np.array(calibration_args[i][1])) for
     i in range(len(calibration_args))]
    plt.legend(['M=%i'% i if i>0 else 'Perfect Calibration' for i in range(0,len(ensemble_calibration_args)+1)],loc="lower center",bbox_to_anchor=(0.5, -0.45),fancybox=True, shadow=True, ncol=4)
    plt.grid()
    plt.xlabel('Confidence (max prob)')
    plt.ylabel('Confidence - Accuracy')
    plt.yticks(np.arange(-0.4, 0.5, step=0.1))
    plt.title('Reliability diagram: SWA solutions', size=15, pad=10)
    plt.savefig('Reliability_diagram_SWA', bbox_inches='tight', dpi=600)
    plt.close()

    plt.axhline(y=0, xmin=0.0, color='black', linestyle='dotted', linewidth=4)
    [plt.plot(ensemble_calibration_args[i][0],
              np.array(ensemble_calibration_args[i][2]) - np.array(ensemble_calibration_args[i][1])) for i in
     range(len(ensemble_calibration_args))]
    # plt.legend(['M=%i'% i if i>0 else 'Perfect Calibration' for i in range(0,len(ensemble_calibration_args)+1)],loc="lower center",bbox_to_anchor=(0.5, -0.45),fancybox=True, shadow=True, ncol=4)
    plt.grid()
    plt.xlabel('Confidence (max prob)')
    plt.ylabel('Confidence - Accuracy')
    plt.title('Reliability diagram: Multi-SWA', size=15, pad=10)
    plt.yticks(np.arange(-0.4, 0.5, step=0.1))
    plt.savefig('Reliability_diagram_Multi-SWA', bbox_inches='tight', dpi=600)
    plt.close()

    plt.grid()
    plt.plot(np.arange(1, 11), [ensemble_accuracies[i][0] for i in range(len(ensemble_accuracies))], color='green')
    plt.axhline(y=np.mean([model_accuracies[i][0] for i in range(len(model_accuracies))]), xmin=0, color='grey',
                ls='dashed', lw=3)
    plt.scatter(np.arange(1, 11), [model_accuracies[i][0] for i in range(len(model_accuracies))], color='black',
                ls='--')
    plt.scatter(np.arange(1, 11), [ensemble_accuracies[i][0] for i in range(len(ensemble_accuracies))], marker='s',
                color='green')
    plt.title('Multi-SWA classification accuracy \n wrt. ensemble size (M)', size=16)
    plt.legend(['Multi-SWA', 'Average SWA accuracy', 'Individual SWA acc.'], bbox_to_anchor=(1.15, -0.15),
               fancybox=True, shadow=True, ncol=3)
    plt.xlabel('Ensemble size (M)')
    plt.ylabel('Acc.')
    plt.yticks(np.arange(0.80, 0.90, 0.01))
    plt.savefig('Multi-SWA_acc', bbox_inches='tight', dpi=600)
    plt.close()

    plt.grid()
    plt.plot(np.arange(1, 11), ensemble_calibration_errors, linewidth=2)
    plt.scatter(np.arange(1, 11), [calibration_errors[i][0] for i in range(len(model_accuracies))], color='sienna',
                ls='--')
    plt.scatter(np.arange(1, 11), [calibration_errors[i][1] for i in range(len(model_accuracies))], color='grey',
                ls='--')
    plt.scatter(np.arange(1, 11), [ensemble_calibration_errors[i][0] for i in range(len(ensemble_calibration_errors))],
                marker='s')
    plt.scatter(np.arange(1, 11), [ensemble_calibration_errors[i][1] for i in range(len(ensemble_calibration_errors))],
                marker='s')
    plt.axhline(y=np.mean([calibration_errors[i][0] for i in range(len(model_accuracies))]), xmin=0, color='sienna',
                ls='dashed', lw=2)
    plt.axhline(y=np.mean([calibration_errors[i][1] for i in range(len(model_accuracies))]), xmin=0, color='grey',
                ls='dashed', lw=2)
    plt.title('Multi-SWA calibration wrt. ensemble size (M)', size=16)
    plt.legend(
        ['Multi-SWA ECE', 'Multi-SWA MCE', 'Average SWA ECE', 'Moving average SWA MCE', 'Individual SWA ECE',
         'Individual model MCE'], bbox_to_anchor=(1.15, -0.15), fancybox=True, shadow=True, ncol=3)
    plt.xlabel('Ensemble size (M)')
    plt.ylabel('Calibration error')
    plt.savefig('Multi-SWA_calibration', bbox_inches='tight', dpi=600)
    plt.close()
def swagDiag():
    from models import Net
    from testFunctions import testModel, testCalibrationBins, testCalibrationError
    from dataloader import get_data, resetSeed
    from ensemble import makeEnsemblePrediction
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    swa_parameters = ['SWA_parameters_model_%i' % i for i in range(69, 79)]
    swag_diagonals = ['SWAG_Diagonal_model_%i' % i for i in range(69, 79)]

    swa_model = Net()
    train, valid, test = get_data()

    # these are used in multi-SWAG
    swag_prediction_probabilities = []
    swag_pred_probability = []

    model_accuracies = []
    calibration_args = []
    calibration_errors = []

    for m in range(len(swa_parameters)):
        swa_weights = torch.load(swa_parameters[m])
        swag_diagonal = torch.load(swag_diagonals[m])

        for draw in range(29):
            torch.nn.utils.vector_to_parameters(torch.normal(swa_weights, abs(swag_diagonal)), swa_model.parameters())
            torch.nn.utils.parameters_to_vector(swa_model.parameters())
            probs, _, labels = testModel(swa_model, test)
            resetSeed(draw)
            swag_pred_probability += [probs]

        # bayesian averaging for models
        swag_probabilities, swag_ensemble_max_probability, swag_ensemble_prediction = makeEnsemblePrediction(
            swag_pred_probability)
        swag_pred_probability = []

        # bayesian for multiswa
        swag_prediction_probabilities += [swag_probabilities]

        model_accuracy = accuracy_score(labels, swag_ensemble_prediction)
        error = 1.96 * np.sqrt((model_accuracy * (1 - model_accuracy)) / len(labels))
        model_accuracies += [(model_accuracy, error)]

        calibration_arg = testCalibrationBins(swag_probabilities, swag_ensemble_prediction, labels)
        calibration_args += [calibration_arg]
        calibration_errors += [testCalibrationError(*calibration_arg)]

    ensemble_accuracies = []
    ensemble_calibration_args = []
    ensemble_calibration_errors = []

    for ensemble_size in range(1, len(swa_parameters) + 1):
        # accuracies
        ensemble_accuracy = accuracy_score(labels, makeEnsemblePrediction(
            swag_prediction_probabilities[:ensemble_size])[2])
        ensemble_error = 1.96 * np.sqrt((ensemble_accuracy * (1 - ensemble_accuracy)) / len(labels))
        ensemble_accuracies += [(ensemble_accuracy, ensemble_error)]

        # calibration errors
        ensemble_calibration_args += [
            testCalibrationBins(makeEnsemblePrediction(swag_prediction_probabilities[:ensemble_size])[0],
                                makeEnsemblePrediction(swag_prediction_probabilities[:ensemble_size])[2],
                                labels)]
        ensemble_calibration_errors += [testCalibrationError(*ensemble_calibration_args[ensemble_size - 1])]

    torch.save(model_accuracies, 'SWAG_model_accuracies')
    torch.save(calibration_args, 'SWAG_calibration_args')
    torch.save(calibration_errors, 'SWAG_model_errors')

    torch.save(ensemble_accuracies, 'Multi_SWAG_model_accuracies')
    torch.save(ensemble_calibration_args, 'Multi_SWAG_calibration_args')
    torch.save(ensemble_calibration_errors, 'Multi_SWAG_model_errors')

    plt.axhline(y=0, xmin=0.0, color='black', linestyle='dotted', linewidth=3)
    [plt.plot(ensemble_calibration_args[i][0], np.array(calibration_args[i][2]) - np.array(calibration_args[i][1])) for
     i in range(len(calibration_args))]
    plt.legend(
        ['Model %i' % i if i > 0 else 'Perfect Calibration' for i in range(0, len(ensemble_calibration_args) + 1)],
        loc="lower center", bbox_to_anchor=(0.475, -0.35), fancybox=True, shadow=True, ncol=4)
    plt.grid()
    plt.xlabel('Confidence (max prob)')
    plt.ylabel('Confidence - Accuracy')
    plt.yticks(np.arange(-0.4, 0.5, step=0.1))
    plt.title('Reliability diagram: SWAG solutions', size=15, pad=20)
    plt.savefig('Reliability_diagram_SWAG', bbox_inches='tight', dpi=600)
    plt.close()

    plt.axhline(y=0, xmin=0.0, color='black', linestyle='dotted', linewidth=4)
    [plt.plot(ensemble_calibration_args[i][0],
              np.array(ensemble_calibration_args[i][2]) - np.array(ensemble_calibration_args[i][1])) for i in
     range(len(ensemble_calibration_args))]
    plt.legend(['M=%i' % i if i > 0 else 'Perfect Calibration' for i in range(0, len(ensemble_calibration_args) + 1)],
               loc="lower center", bbox_to_anchor=(0.475, -0.35), fancybox=True, shadow=True, ncol=4)
    plt.grid()
    plt.xlabel('Confidence (max prob)')
    plt.ylabel('Confidence - Accuracy')
    plt.title('Reliability diagram: Multi-SWAG (Diagonal)', size=15, pad=10)
    plt.yticks(np.arange(-0.4, 0.5, step=0.1))
    plt.savefig('Reliability_diagram_Multi-SWAG', bbox_inches='tight', dpi=600)
    plt.close()

    plt.grid()
    plt.plot(np.arange(1, len(swa_parameters) + 1),
             [ensemble_accuracies[i][0] for i in range(len(ensemble_accuracies))], color='darkorange')
    plt.axhline(y=np.mean([model_accuracies[i][0] for i in range(len(model_accuracies))]), xmin=0, color='grey',
                ls='dashed', lw=3)
    plt.scatter(np.arange(1, len(swa_parameters) + 1), [model_accuracies[i][0] for i in range(len(model_accuracies))],
                color='black',
                ls='--')
    plt.scatter(np.arange(1, len(swa_parameters) + 1),
                [ensemble_accuracies[i][0] for i in range(len(ensemble_accuracies))], marker='s',
                color='darkorange')
    plt.title('Multi-SWAG Diagonal classification accuracy \n wrt. ensemble size (M)', size=16)
    plt.legend(['Multi-SWAG', 'Average SWAG accuracy', 'Individual SWAG acc.'], bbox_to_anchor=(1.15, -0.15),
               fancybox=True, shadow=True, ncol=3)
    plt.xlabel('Ensemble size (M)')
    plt.ylabel('Acc.')
    plt.yticks(np.arange(0.80, 0.90, 0.01))
    plt.savefig('Multi-SWA_acc', bbox_inches='tight', dpi=600)
    plt.close()

    plt.grid()
    plt.plot(np.arange(1, len(swa_parameters) + 1),
             [ensemble_accuracies[i][0] for i in range(len(ensemble_accuracies))], color='darkorange')
    plt.axhline(y=np.mean([model_accuracies[i][0] for i in range(len(model_accuracies))]), xmin=0, color='grey',
                ls='dashed', lw=3)
    plt.scatter(np.arange(1, len(swa_parameters) + 1), [model_accuracies[i][0] for i in range(len(model_accuracies))],
                color='black',
                ls='--')
    plt.scatter(np.arange(1, len(swa_parameters) + 1),
                [ensemble_accuracies[i][0] for i in range(len(ensemble_accuracies))], marker='s',
                color='darkorange')
    plt.title('Multi-SWAG Diagonal classification accuracy \n wrt. ensemble size (M)', size=16)
    plt.legend(['Multi-SWAG', 'Average SWAG accuracy', 'Individual SWAG acc.'], bbox_to_anchor=(1.15, -0.15),
               fancybox=True, shadow=True, ncol=3)
    plt.xlabel('Ensemble size (M)')
    plt.ylabel('Acc.')
    plt.yticks(np.arange(0.80, 0.90, 0.01))
    plt.savefig('Multi-SWA_acc', bbox_inches='tight', dpi=600)
    plt.close()

    plt.grid()
    plt.plot(np.arange(1, 11), ensemble_calibration_errors, linewidth=2)
    plt.scatter(np.arange(1, 11), [calibration_errors[i][0] for i in range(len(model_accuracies))], color='sienna',
                ls='--')
    plt.scatter(np.arange(1, 11), [calibration_errors[i][1] for i in range(len(model_accuracies))], color='grey',
                ls='--')
    plt.scatter(np.arange(1, 11), [ensemble_calibration_errors[i][0] for i in range(len(ensemble_calibration_errors))],
                marker='s')
    plt.scatter(np.arange(1, 11), [ensemble_calibration_errors[i][1] for i in range(len(ensemble_calibration_errors))],
                marker='s')
    plt.axhline(y=np.mean([calibration_errors[i][0] for i in range(len(model_accuracies))]), xmin=0, color='sienna',
                ls='dashed', lw=2)
    plt.axhline(y=np.mean([calibration_errors[i][1] for i in range(len(model_accuracies))]), xmin=0, color='grey',
                ls='dashed', lw=2)
    plt.title('Multi-SWAG Diagonal calibration wrt. ensemble size (M)', size=16)
    plt.legend(
        ['Multi-SWAG ECE', 'Multi-SWAG MCE', 'Average SWAG ECE', 'Average SWAG MCE', 'Individual SWAG ECE',
         'Individual SWAG MCE'], bbox_to_anchor=(1.15, -0.15), fancybox=True, shadow=True, ncol=3)
    plt.xlabel('Ensemble size (M)')
    plt.ylabel('Calibration error')
    plt.savefig('Multi-SWAG_calibration', bbox_inches='tight', dpi=600)
    plt.close()












