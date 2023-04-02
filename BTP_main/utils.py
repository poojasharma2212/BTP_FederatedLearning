import torch
import math


def averageModels(global_model, clients, snr_value, Ps,alpha,K_clients):
    client_models = [clients[i]['model'] for i in range(len(clients))]
    
    samples = [clients[i]['samples'] for i in range(len(clients))]

    global_dict = global_model.state_dict()

    snr = snr_value
    # print("SNR==", snr)
    snr_val = 10**(snr/10)
    std = math.sqrt(Ps/snr_val)

    for k in global_dict.keys():  
        client_weights = [client_models[i].state_dict()[k].float() * samples[i] for i in range(len(client_models))]
        weighted_sum = torch.stack(client_weights, dim=0).sum(dim=0)
        global_dict[k] = weighted_sum

        # Add Gaussian noise to the global model's parameters
        noise = torch.randn(global_dict[k].shape) * std
        print(noise.size())
        global_dict[k] += noise

    # global_model.load_state_dict(global_dict)
    # for k in global_dict.keys():  # key is CNN layer index and value is layer parameters
    #     # take a weighted average and not average because the clients may not have the same amount of data to train upon
        
    #     global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() * samples[i]  for i in range(len(client_models))], 0).sum(0)
        
    #     # global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() * samples[i]+   torch.randn(client_models[0].state_dict()[k].float().size())*std for i in range(len(client_models))], 0).sum(0) 
        # #take a weighted average and not average because the clients may not have the same amount of data to train upon

    # print(global_dict.size())
    # torch.flatten(global_model)
    global_model.load_state_dict(global_dict)

    return global_model


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