import os, argparse
import torch
from datetime import datetime
from utils.models import get_model_info
from utils.dataset import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_name(dataset_path, clustering, model_name, epochs):
    dataset_path = dataset_path[:-1] if dataset_path[-1] == "/" else dataset_path
    dataset_info = dataset_path.split('/')

    info_idx = -2 if dataset_info[-1] == "train" else -1
    cluster = "-cluster" if clustering else "-noCluster" if dataset_info[info_idx-1] == "poisoned" else ""
    timestamp = datetime.now().strftime('%m%d-%H%M')
    return f"Model_{dataset_info[info_idx]}-{dataset_info[info_idx-1]}{cluster}_{model_name}_{epochs}_{timestamp}"

def execute_training(dataset_path, model_name, epochs, batch_size, clustering=False):
    _, dataloader = load_data(dataset_path, batch_size, clustering)
    model, optim, sched = get_model_info(model_name, operation="train")

    file_name = set_name(dataset_path, clustering, model_name, epochs)

    with open(os.path.join("database/models", file_name + ".txt"), "w") as LOGGER:
        LOGGER.write(f"\nDataset: {dataset_path}")
        LOGGER.write(f"\nClustering: {clustering}")
        LOGGER.write(f"\nModel: {model_name}")
        LOGGER.write(f"\nEpochs: {epochs}")

        net = model().to(device)
        opt = optim(net.parameters())
        sch = sched(opt) if sched is not None else None

        criterion = torch.nn.CrossEntropyLoss()
        train(net, criterion, opt, epochs, dataloader, device, 0, scheduler=sch, log=LOGGER)
    torch.save(net.state_dict(), os.path.join("database/models", file_name + ".pth"))
    return file_name

def train(net, criterion, optimizer, epochs, trainloader, device, past_epochs=0, scheduler=None, log=None):
    net.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += int(labels.size(0))
            correct += int((predicted == labels).sum())

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                log.write('\n[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        log.write('\nAccuracy for epoch %d: %d %%' % (past_epochs + epoch + 1, 100 * correct / total))
        if scheduler is not None:
            scheduler.step()

        # Write to file every 10 epochs
        if epoch % 10 == 9:
            log.flush() ; os.fsync(log.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str, help='Path to the dataset we want to train')
    parser.add_argument("--model", "-m", type=str, help='CNN model to perform clustering', choices=["resnet32", "resnet18", "customnet"], default="resnet32")
    parser.add_argument("--epochs", "-e", required=True, type=int, help='Number of training epochs', default=200)
    parser.add_argument("--batch", "-b", type=int, help='Batch size', default=128)
    parser.add_argument("--cluster", action="store_true", help="Indicates whether to load cleaned samples")
    args = parser.parse_args()
    print(args.dataset, args.model, args.epochs, args.batch, args.cluster)
    execute_training(args.dataset, args.model, args.epochs, args.batch, args.cluster)
