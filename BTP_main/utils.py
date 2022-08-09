import torch
import math


def averageModels(global_model, clients, snr_value, Ps):
    client_models = [clients[i]['model'] for i in range(len(clients))]
    samples = [clients[i]['samples'] for i in range(len(clients))]
    global_dict = global_model.state_dict()

    print('global_dict', global_dict)
    for k in global_dict.keys():  # key is CNN layer index and value is layer parameters
        # take a weighted average and not average because the clients may not have the same amount of data to train upon
        snr = snr_value
        print("SNR==", snr)
        snr_val = 10**(snr/10)
        std = math.sqrt(Ps/snr_val)
        # print("Client", client_models)
        print("-----------------")
        print(std)
        # print(global_dict)
        # noise = torch.randn(global_dict.size())
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float(
        ) * samples[i] for i in range(len(client_models))], 0).sum(0)

    print(global_dict)
    global_model.load_state_dict(global_dict)

    return global_model
