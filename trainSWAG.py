from dataloader import *
from models import *

resetSeed()
master_train, train, valid, test = get_data(True)
baselines = [r"model%i_final" % i for i in range(69,79)]

for model_number, baseline in enumerate(baselines,69):
    pretrained_model = Net()
    pretrained_model.load_state_dict(torch.load(baseline))
    resetSeed()

    torch.backends.cudnn.deterministic = True
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        net = pretrained_model.cuda()
        criterion = criterion.cuda()

    swa_model = copy.deepcopy(net)
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    theta_squared = torch.nn.utils.parameters_to_vector(net.parameters()) ** 2
    D = []

    training_loss = []
    training_loss_swa = []

    training_acc = []
    training_acc_swa = []

    valid_loss_epochs = []
    valid_loss_epochs_swa = []

    valid_acc = []
    valid_acc_swa = []

    T = 1

    for epoch in range(24):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train, 0):
            if i % 100 == 0:
                print('data number %f in epoch %f' % (i, epoch))
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        weights = torch.nn.utils.parameters_to_vector(swa_model.parameters())

        # average model
        new_weights = torch.nn.utils.parameters_to_vector(net.parameters())

        # second moments
        new_theta_squared = new_weights ** 2

        # weighted average of mean
        weights = (T * weights + new_weights) / (T + 1)

        # weighted average of second moment
        theta_squared = (T * theta_squared + new_theta_squared) / (T + 1)

        torch.nn.utils.vector_to_parameters(weights, swa_model.parameters())

        # second moment of SWA averages
        theta_squared_SWA = weights ** 2

        # SWAG-Diagonal
        swag_diag = theta_squared-theta_squared_SWA

        # Low-rank
        D += [(new_weights - weights).tolist()]

        T += 1

        net.eval()
        swa_model.eval()

        valid_loss = 0.0
        valid_loss_swa = 0.0

        valid_correct = 0
        valid_total = 0

        valid_total_swa = 0
        valid_correct_swa = 0

        with torch.no_grad():
            for val_image, val_label in valid:
                if torch.cuda.is_available():
                    val_image, val_label = val_image.cuda(), val_label.cuda()

                outputs_net = net(val_image)
                loss = criterion(outputs_net, val_label)
                valid_loss += loss.item()

                _, predicted_net = outputs_net.max(1)
                valid_total += val_label.size(0)
                valid_correct += (predicted_net == val_label).sum().item()

                outputs_swa = swa_model(val_image)
                # Find the Loss
                loss_swa = criterion(outputs_swa, val_label)
                # Calculate Loss
                valid_loss_swa += loss_swa.item()

                _, predicted_net_swa = outputs_swa.max(1)
                valid_total_swa += val_label.size(0)
                valid_correct_swa += (predicted_net_swa == val_label).sum().item()

        valid_acc_swa += [100 * valid_correct_swa / valid_total_swa]
        valid_loss_epochs_swa += [valid_loss_swa / len(valid)]

        training_loss += [running_loss / len(train)]
        valid_loss_epochs += [valid_loss / len(valid)]

        training_acc += [100 * correct / total]
        valid_acc += [100 * valid_correct / valid_total]


        print('epoch %f SGD train loss %f ' % (int(epoch), training_loss[epoch]))
        print('epoch %f SGD training accuracy %f ' % (int(epoch), training_acc[epoch]))

        print('epoch %f SGD validation loss %f ' % (int(epoch), valid_loss_epochs[epoch]))
        print('epoch %f SGD validation accuracy %f ' % (int(epoch), valid_acc[epoch]))

        print('epoch %f SWA validation loss %f ' % (int(epoch), valid_loss_swa / len(valid)))
        print('epoch %f SWA validation accuracy %f ' % (int(epoch), valid_acc_swa[epoch]))

        net.train()
        print('done with epoch %f' % epoch)

        torch.save(net.state_dict(), r'model_%i_SGD_net_epoch_%i' % (int(model_number),int(epoch)))
        torch.save(swa_model.state_dict(), r'model_%i_SWA_net_epoch_%i' % (int(model_number),int(epoch)))

    torch.save(torch.tensor(valid_loss_epochs), r'valid_loss_model_%i_SGD' % int(model_number))
    torch.save(torch.tensor(valid_loss_epochs_swa), r'valid_loss_model_%i_SWA' % int(model_number))

    torch.save(torch.tensor(valid_acc), r'valid_acc_model_%i_SGD' % int(model_number))
    torch.save(torch.tensor(valid_acc_swa), r'valid_acc_model_%i_SWA' % int(model_number))

    torch.save(net.state_dict(), r'model_%i_SGD_final' % int(model_number))
    torch.save(swa_model.state_dict(), r'model_%i_SWA_final' % int(model_number))

    torch.save(torch.nn.utils.parameters_to_vector(swa_model.parameters()),r'SWA_parameters_model_%i' % int(model_number))
    torch.save(swag_diag,r'SWAG_Diagonal_model_%i' % int(model_number))

