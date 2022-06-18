from dataloader import resetSeed
import pandas as pd

def testModel(model,test,softmax = True):
    import torch
    resetSeed()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = model.cuda()

    all_probabilites = []
    all_predictions = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test:
            # extremely important
            torch.manual_seed(69)
            torch.cuda.manual_seed_all(69)

            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            all_labels += [labels[0].tolist()]
            outputs = model(images)

            if softmax == True:
                outputs = torch.softmax(outputs, dim=1)

            #probabilites
            all_probabilites += [outputs.tolist()[0]]
            _, predicted = torch.max(outputs.data, 1)
            all_predictions += [predicted.tolist()[0]]

            # accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %f %%' % (len(test), 100 * correct / total))

    return all_probabilites, all_predictions, all_labels

def testSWAG_Diagonal(net,swa_parameters,SWAG_diagonal,S):
    probabilities_swag = 0
    for models in range(30):
        swag = net
        swag_parameter = torch.normal(swa_parameters, SWAG_diagonal)

        # update swag
        torch.nn.utils.vector_to_parameters(swag_parameter, swag.parameters())

        probabilities_swag += (1 / 30) * torch.tensor(testModel(swag, test)[0])

    # TODO: return most argmax


def testCalibrationBins(probabilities, predictions, labels):
    import numpy as np
    import pandas as pd
    max_probabilities = [probabilities[i][predictions[i]] for i in range(len(probabilities))]

    # since we can never have a max-predictions probability of < 100/7 we start binning at 0.2
    bins = np.linspace(0.2, 1, 9)
    binned = np.digitize(max_probabilities, bins, right=True)

    calibration = pd.DataFrame(list(zip(probabilities, max_probabilities, binned, predictions, labels)),
                               columns=['probabilities', 'max_probabilities', 'binned', 'predictions', 'labels'])

    bin_n = [len(calibration[calibration.binned == bin]) for bin in np.arange(len(bins))]
    bin_accuracy = [
        sum(calibration[calibration.binned == bin].predictions == calibration[calibration.binned == bin].labels) / len(
            calibration[calibration.binned == bin]) if len(calibration[calibration.binned == bin]) > 0 else 0 for bin in
        np.arange(len(bin_n))]
    bin_conf = [calibration[calibration.binned == bin].max_probabilities.mean() if pd.isnull(
        calibration[calibration.binned == bin].max_probabilities.mean()) == False else 0 for bin in
                np.arange(len(bin_n))]

    return list(bins), bin_accuracy, bin_conf, bin_n


def testCalibrationError(bins, bin_accuracy, bin_conf, bin_n):
    import numpy as np
    ECE = sum(
        [(bin_n[bin] / sum(bin_n)) * abs(bin_accuracy[bin] - bin_conf[bin]) for bin in np.arange(len(bin_accuracy))])

    MCE = max(abs(np.array(bin_accuracy) - np.array(bin_conf)))
    return ECE, MCE


def testMcNemar(predictions_1, predictions_2, labels):
    from statsmodels.stats.contingency_tables import mcnemar
    import torch

    test1 = torch.tensor(predictions_1) == torch.tensor(labels)
    test2 = torch.tensor(predictions_2) == torch.tensor(labels)

    yesyes = sum([test1[i] == True and test2[i] == True for i in range(len(test1))])
    yesno = sum(test1 > test2)
    nono = sum([test1[i] == False and test2[i] == False for i in range(len(test1))])
    noyes = sum(test1 < test2)

    contingency_table = np.array([[yesyes, yesno], [noyes, nono]])

    return mcnemar(contingency_table, exact=True).pvalue


def testF1(labels, predictions):
    from sklearn.metrics import f1_score
    import numpy as np
    return f1_score(labels, predictions, labels=np.unique(labels), average='weighted')






