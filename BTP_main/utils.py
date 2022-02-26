def averageModels(global_model, clients):
    client_models = [clients[i]['model'] for i in range(len(clients))]
    samples = [clients[i]['samples'] for i in range(len(clients))]
    global_dict = global_model.state_dict()
        
    for k in global_dict.keys(): #key is CNN layer index and value is layer parameters
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() * samples[i] for i in range(len(client_models))], 0).sum(0) #take a weighted average and not average because the clients may not have the same amount of data to train upon
                
    global_model.load_state_dict(global_dict)
    return global_model

