# for client in clients:
#       train(args, client, device, client['optim'])
#       test(model, client ,client['mnist_testset'])
    
# thats all we need to do XD
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())
# batchSize = 128
# train_loader = torch.utils.data.DataLoader(dataset = mnist_trainset,
#                                            batch_size=batchSize,
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset = mnist_testset,
#                                            batch_size=batchSize,
#                                            shuffle=False)

# def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
#  # raise NotImplementedError("Subclasses should implement this!")
#   train_losses = np.zeros(epochs)
#   test_losses = np.zeros(epochs)

#   for it in range(epochs):
#     t0 = datetime.now()
#     train_loss = []
#     for inputs, targets in train_loader:
#       inputs, targets = inputs.to(device), targets.to(device)  #moving data to GPU

#       optimizer.zero_grad() # set parameter gradient to zero

#       outputs = model(inputs)  # forward pass
#       loss = criterion(outputs, targets)

#       loss.backward()  #backward and optimize
#       optimizer.step()

#       train_loss.append(loss.item())
#     train_loss = np.mean(train_loss)
    
#     test_loss = []
#     for inputs, targets in test_loader:
#       inputs, targets = inputs.to(device), targets.to(device)
#       outputs = model(inputs)
#       loss = criterion(outputs, targets)
#       test_loss.append(loss.item())
#     test_loss = np.mean(test_loss)

#     train_losses[it] = train_loss
#     test_losses[it] = test_loss

#     dt = datetime.now() - t0

#     print(f'Epoch{it+1}/{epochs}, Train Loss: {train_loss: .4f}, \
#     Test Loss: {test_loss:.4f}, Duration: {dt}')

#   return train_losses, test_losses

      

# train_losses, test_losses = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs=15)
# plt.plot(train_losses, label='train loss')
# plt.plot(test_losses, label='test loss')
# plt.legend()
# plt.show()


# n_correct = 0.
# n_total = 0.

# for inputs, targets in train_loader:
#   inputs, targets = inputs.to(device), targets.to(device) # moving data to GPU

#   outputs = model(inputs)

#   _, predictions = torch.max(outputs, 1) 

#   n_correct  = n_correct + (predictions==targets).sum().item()
#   n_total = n_total + targets.shape[0]

# train_acc = n_correct / n_total

# n_correct = 0.
# n_total = 0.

# for inputs, targets in test_loader:
#   inputs, targets = inputs.to(device), targets.to(device) # moving data to GPU

#   outputs = model(inputs)

#   _, predictions = torch.max(outputs, 1) 

#   n_correct  = n_correct + (predictions==targets).sum().item()
#   n_total = n_total + targets.shape[0]

# test_acc = n_correct / n_total


# print(f"Train acc: {train_acc: .4f}, Test acc: {test_acc: .4f}") 