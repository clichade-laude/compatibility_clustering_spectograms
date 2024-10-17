import os
import torch

from utils.models import get_model_info
from utils.dataset import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def execute_testing(dataset, model_path, batch_size):
    dataset_path = os.path.join("database/original", dataset, "test")
    _, dataloader = load_data(dataset_path, batch_size)

    model, _, _ = get_model_info(model_path.split('/')[-1].split('_')[2], operation="train")
    net = model().to(device)
    net.load_state_dict(torch.load(model_path))

    accuracy, misclassified = test(net, dataloader, device, 0)
    ms = sum(misclassified) / len(misclassified)
    print(f"Accuracy: {accuracy}")
    print(f"Missclassified: {ms}")


def test(net, testloader, device, source, target=None):
    net.eval()
    correct = 0
    total = 0
    target_misclassified = []

    with torch.no_grad():
        last_idx = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if target is not None:
                # poisoned test set
                end_idx = last_idx + len(labels)
                true_labels = \
                        torch.Tensor(testloader.dataset.true_targets[last_idx:end_idx])
                true_labels = true_labels.to(device)
                source_labels = (true_labels == source)
                last_idx = end_idx

                # which elements of the source class are incorrect classified as target class
                wrong_class = predicted == target
                target_misclassified.extend(wrong_class[source_labels])

                labels = true_labels
            else:
                # clean test set
                # which elements of the source class are correct classified as source class
                correct_class = predicted == source
                source_labels = (labels == source)
                target_misclassified.extend(correct_class[source_labels])

            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy, target_misclassified

execute_testing("cifar", "database/models/Model_cifar-poisoned-cluster_resnet32_1_1018-1018.pth", 64)