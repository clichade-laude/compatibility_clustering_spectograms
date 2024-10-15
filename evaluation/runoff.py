import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


import re
import numpy as np
from os import remove, fsync
from os.path import join

PATIENCE = 200
LR = 0.0005
log = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

def process_epoch(split_loader, isTrain, model):
    running_loss = 0.
    for i, data in enumerate(split_loader):
        ## Load batch data in device
        inputs, labels = data[0].to(device), data[1].to(device)
        if isTrain:
            optimizer.zero_grad()
        ## Make predictions for current batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        if isTrain:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
    return running_loss/len(split_loader)


def train(model, epochs: int, train_loader: DataLoader, valid_loader: DataLoader, save_folder: str, partition: int = 1):
    global criterion, optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_epoch, best_vloss = -1, 1_000_000.

    for epoch in range(int(epochs)):
        log.write(f"\n\tEpoch {epoch}")

        model.train()
        avg_loss = process_epoch(train_loader, True, model)

        model.eval()
        with torch.no_grad():
            avg_vloss = process_epoch(valid_loader, False, model)
        log.write("\n\tLOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            ## Model path base
            model_path = join(save_folder, f"BestModel-Part_{partition}-Epoch_")
            ## Save best model and remove previous
            torch.save(model.state_dict(), model_path+str(epoch))
            if epoch != 0:
                remove(model_path+str(best_epoch))
            ## Update values
            best_vloss = avg_vloss
            best_epoch = epoch
        else:
            if epoch - best_epoch > PATIENCE:
                log.write(f"\n\t{PATIENCE} epochs without improvement. Stopping training.")
                break

        ## Write to file every 10 epochs
        if epoch % 10 == 9:
            log.flush() ; fsync(log.fileno())

    log.write("\n\tFinished Training")
    log.flush() ; fsync(log.fileno())

def test(model, test_loader: DataLoader):
    model.eval()
    correct, total = 0, 0
    all_outputs = torch.tensor([], device=device)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            all_outputs = torch.cat((all_outputs, outputs), 0)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    log.write(f"\n\tTest Accuracy: {accuracy:.2f}%")
    return all_outputs, accuracy

