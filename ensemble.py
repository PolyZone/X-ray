def DeepEnsembleTest(ensemble, softmax=True):
    import torch
    from models import Net
    from dataloader import get_data,resetSeed
    net = Net()
    train,valid,test = get_data()
    resetSeed()

    device = torch.device("cuda")
    net.to(device)

    ensemble_models_probabilities = []
    ensemble_models_ensemble_predictions = []

    for model in ensemble:
        net.load_state_dict(torch.load(model))
        net.to(device)

        all_probabilites = []
        all_predictions = []
        all_labels = []
        model_test_outputs = []

        correct = 0
        total = 0

        net.eval()
        with torch.no_grad():
            for data in test:

                # extremely important
                torch.manual_seed(69)
                torch.cuda.manual_seed_all(69)

                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                all_labels += [labels[0].tolist()]
                outputs = net(images)

                if softmax == True:
                    outputs = torch.softmax(outputs, dim=1)

                all_probabilites += [outputs.tolist()[0]]
                _, predicted = torch.max(outputs.data, 1)
                all_predictions += [predicted.tolist()[0]]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print('Accuracy of the network %s on the %d test images: %d %%' % (model,len(test),
        #   100 * correct / total))

        ensemble_models_probabilities += [all_probabilites]
        # print(all_probabilites[0])
        # print(all_predictions[0])

        ensemble_models_ensemble_predictions += [all_predictions]

    return ensemble_models_probabilities, ensemble_models_ensemble_predictions, all_labels
def makeEnsemblePrediction(ensemble_models_probabilities):
    import numpy as np
    import torch
    probabilities = sum(np.array([np.array(predictions) for predictions in ensemble_models_probabilities])) / len(
        ensemble_models_probabilities)
    ensemble_probabilities = probabilities
    ensemble_max_probability = torch.max(torch.Tensor(probabilities), 1).values
    ensemble_prediction = torch.max(torch.Tensor(probabilities), 1).indices

    return ensemble_probabilities, ensemble_max_probability, ensemble_prediction

