import os, argparse, pickle
import torch

from utils.models import get_model_info
from utils.dataset import load_data
from utils.mqtt import connect_node, publish_mqtt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def execute_testing(dataset, orig_dataset, batch_size, test_folder, poison_info):
    test_path = os.path.join("database", "results", test_folder)
    ## Obtain model path, its info and load it
    model_file = [file for file in os.listdir(test_path) if file.endswith('.pth')][0]
    model, _, _ = get_model_info(model_file.split('_')[5], operation="train")
    net = model().to(device)
    net.load_state_dict(torch.load(os.path.join(test_path, model_file)))

    ## Load dataloader for poison images and perform testing
    if poison_info:
        with open(f'{poison_info}', 'rb') as f:
            params = pickle.load(f)
        _, poisonloader = load_data(os.path.join("database/poisoned", dataset, "test"), batch_size)
        poisonloader.dataset.obtain_poisoned()
        poison_accuracy, source_poison = test(net, poisonloader, device, params["source"], params["target"])

    ## Load dataloader for clean images and perform testing
    _, cleanloader = load_data(os.path.join("database/original", orig_dataset, "test"), batch_size)
    source_class = 0 if not poison_info else params["source"]
    clean_accuracy, source_clean = test(net, cleanloader, device, source_class)

    with open(os.path.join(test_path, test_folder + "_test.txt"), "w") as logger:
        p = sum(source_clean) / len(source_clean)
        logger.write(f"\nAccuracy on clean testset: {clean_accuracy}")
        if poison_info:
            logger.write(f"\nClean correct classification: {p}")
            p = sum(source_poison) / len(source_poison)
            logger.write(f"\nAccuracy on poison testset: {poison_accuracy}")
            logger.write(f"\nPoison misclassification: {p}")
            poison_misclassification = [p and c for p, c in zip(source_poison, source_clean)]
            p = sum(poison_misclassification) / len(poison_misclassification)
            logger.write(f"\nTargeted misclassification: {p}")


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

def on_message(client, userdata, msg):
    print("Node: testing | Executing", flush=True)
    import json
    params = json.loads(msg.payload)
    execute_testing(params["dataset"], params["orig_dataset"], params["batch"], params["folder"], params["poison"])

    publish_mqtt(client, "control", node=userdata)

if __name__ == "__main__":
    connect_node("test", on_message)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", "-d", required=True, type=str, help='Name of the dataset to test')
    # parser.add_argument("--model", "-m", required=True, type=str, help='Path to the model we wanna test')
    # parser.add_argument("--batch", "-b", type=int, help='Batch size', default=128)
    # args = parser.parse_args()
    # print(args.dataset, args.model, args.batch)
    # execute_testing(args.dataset, args.model, args.batch)
    # execute_testing("cifar_0-2_0.2", "cifar", 128, "241220-0732__cifar_resnet32_E1_B128__backdoor_0-2_0.2_1-32__NoCluster", "database/backdoor/backdoor_0-2_0.2_1-32.pickle")