import torch
import math


def averageModels(global_model, clients, snr_value, Ps):
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
        # print("Client", client_models)
        print("-----------------")
        # print(std)
        # print(global_dict)
        # noise = torch.randn(global_dict.size())
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float(
        ) * samples[i] for i in range(len(client_models))], 0).sum(0)

    # print(global_dict)
    global_model.load_state_dict(global_dict)

    y_out = global_model.conv1.weight
    x = torch.flatten(y_out)
    xTx = 0
    # should I use here also normalise ??
    for i in range(list(x.size())[0]):
        xTx = xTx + x[i]*x[i]

    print('-----------')
    print("newModel Normalised value : ", xTx)
    print(xTx)
    # if(xTx <= Ps):
    Ps = Ps/xTx
    y_out = y_out*math.sqrt(Ps)
    # else:
    # y_out = y_out*math.sqrt(Ps)/((h)*xTx)
    noise = torch.randn(y_out.size())
    y_out = y_out + noise*std
    y_out = y_out/(math.sqrt(Ps))
    # y_out = y_out.real

    global_model.conv1.weight.data = y_out

    y_out = global_model.conv2.weight
    yy = torch.flatten(y_out)
    yTy = 0
    for i in range(list(yy.size())[0]):
        yTy = yTy + yy[i]*yy[i]

    print('-----------')
    print("xTTTTTTTTTTTTx: ", yTy)
    print(yTy)
    Ps = Ps/yTy
    # if(yTy <= Ps):
    y_out = y_out*math.sqrt(Ps)
    # else:
    # y_out = y_out*math.sqrt(Ps)/((h)*yTy)
    noise = torch.randn(y_out.size())
    y_out = y_out + noise*std
    y_out = y_out/(math.sqrt(Ps))

    global_model.conv2.weight.data = y_out

    return global_model
