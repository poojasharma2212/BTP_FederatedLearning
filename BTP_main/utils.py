import torch
import math


def averageModels(global_model, clients, snr_value, Ps,alpha):
    client_models = [clients[i]['model'] for i in range(len(clients))]
    samples = [clients[i]['samples'] for i in range(len(clients))]

    global_dict = global_model.state_dict()

    # print('global_dict', global_dict)
    for k in global_dict.keys():  # key is CNN layer index and value is layer parameters
        # take a weighted average and not average because the clients may not have the same amount of data to train upon
        snr = snr_value
        # print("SNR==", snr)
        snr_val = 10**(snr/10)
        std = math.sqrt(Ps/snr_val)
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float(
        ) * samples[i] for i in range(len(client_models))], 0).sum(0)

    # print(global_dict)
    # torch.flatten(global_model)
    global_model.load_state_dict(global_dict)


    # y_out = global_model.conv1.weight
    # x = torch.flatten(y_out)
    # xTx = 0
    # # should I use here also normalise ??
    # for i in range(list(x.size())[0]):
    #     xTx = xTx + x[i]*x[i]

    # print('-----------')
    # print("newModel Normalised value : ", xTx)
    # print(xTx)
    # # y_out = y_out*math.sqrt(Ps)
    # noise = torch.randn(y_out.size())
    # y_out = y_out + noise*std
    # y_out = y_out/(math.sqrt(Ps)*27)

    # global_model.conv1.weight.data = y_out

    # y_out = global_model.conv2.weight
    # # y_out = y_out*math.sqrt(Ps)
    # noise = torch.randn(y_out.size())
    # y_out = y_out + noise*std
    # y_out = y_out/(math.sqrt(Ps)*27)

    # global_model.conv2.weight.data = y_out

    # print(global_model)
    return global_model
