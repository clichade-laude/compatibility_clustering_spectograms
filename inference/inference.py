import os, argparse, pickle, time, json
import torch, numpy as np
from prometheus_client import start_http_server, Gauge

import sys
sys.path.append("/home/laude/compatibility_clustering_spectograms/")

from utils.models import get_model_info
from utils.dataset import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric_accuracy = Gauge("inference_accuracy", "Overall model accuracy", ["model", "cluster", "poison"])
metric_sourceAcc = Gauge("inference_source_accuracy", "Source class accuracy", ["model", "cluster", "poison"])
metric_targetMiss = Gauge("inference_target_missclass", "Target class missclassification", ["model", "cluster", "poison"])


def get_datasets(dataset):
    dataset_clean = dataset.split("_")[0]
    cleanset, _ = load_data(os.path.join("database/original", dataset_clean, "test"))
    poisonset, _ = load_data(os.path.join("database/poisoned", dataset, "test"))
    poisonset.obtain_poisoned()

    return cleanset, poisonset

def get_models(models):
    net_lst, net_info = [], []
    for model in models:
        model_info = model.replace(".pth", "").split("__")
        net_name = model_info[2].split("_")[1]
        net = get_model_info(net_name, operation="train")[0]
        net = net().to(device)
        net.load_state_dict(torch.load(os.path.join("database/models", model), map_location=device))
        net_lst.append(net)
        net_info.append((model_info[-2] != "NoPoison", model_info[-1] != "NoCluster"))
    return net_lst, net_info

def initialize_server():
    port = 8000
    start_http_server(port)
    server_entry = [{
        "targets": [f"inference_container:{port}"],
        "labels": {
            "job": "inference_container"
        }
    }]
    update_server(server_entry)

def update_server(data=[]):
    with open('/etc/prometheus/targets/targets.json', 'w') as f:
        json.dump(data, f, indent=4)    

def set_metric(metric: Gauge, model, poison, cluster, value):
    metric.labels(model=model, poison=poison, cluster=cluster).set(value)
    # print(f"{metric._documentation}: {value}", flush=True)

def execute_inference(dataset, poison_info, *models):
    initialize_server()
    ## Load datasets, models and poison info
    cleanset, poisonset = get_datasets(dataset)
    poisonset.obtain_poisoned()

    nets_lst, nets_info = get_models(models)
    with open(os.path.join("database/backdoor", poison_info), 'rb') as f:
        params = pickle.load(f)

    models_stats = np.zeros((len(models), 3)) ## Correctly predicted, source clean predicted, source bad predicted

    ## Loop and enumerate images in clean and poisoned dataset. Do it randonly to more visual results
    total_source = 0
    for n_imgs, img in enumerate(np.random.permutation(cleanset.imgs_names)):
        clean_idx = np.where(cleanset.imgs_names == img)[0][0]
        clean_img, true_label = cleanset[clean_idx]

        isSource = true_label == params['source']
        if isSource:
            total_source += 1
            poison_idx = np.where(poisonset.imgs_names == img)[0][0]
            poison_img, label = poisonset[poison_idx]

        for nn, net in enumerate(nets_lst):
            ## Check if sample was well predicted, and sum one if so
            wellPredicted = inference(net, clean_img) == true_label
            models_stats[nn][0] += wellPredicted 
            if isSource:
                ## If it was a source class, sum 1 also here
                models_stats[nn][1] += wellPredicted 
                ## Check if target have been predicted on poisonset
                targetPredicted = inference(net, poison_img) == params['target']
                ## Add 1 if source class well predicted on cleanset and target predicted on poisonset
                models_stats[nn][2] += wellPredicted and targetPredicted 
            ## Export obtained data to prometheus
            set_metric(metric_accuracy, models[nn], nets_info[nn][0], nets_info[nn][1], models_stats[nn][0]/(n_imgs+1))
            set_metric(metric_sourceAcc, models[nn], nets_info[nn][0], nets_info[nn][1], models_stats[nn][1]/total_source)
            set_metric(metric_targetMiss, models[nn], nets_info[nn][0], nets_info[nn][1], models_stats[nn][2]/total_source)
        time.sleep(1)
    update_server()


def inference(net, img):
    net.eval()
    with torch.no_grad():
        image = img.to(device)
        image = torch.unsqueeze(image, 0) ## add one dimension to simulate batch
        output = net(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str, help='Name of the dataset to test')
    parser.add_argument("--poison", "-p", required=True, type=str, help='Name of the pickle file with the poison info')
    parser.add_argument("--model", "-m", required=True, type=str, help='Any model we want to inference together', nargs="+")
    args = parser.parse_args()
    print(args)
    execute_inference(args.dataset, args.poison, *args.model)
    # execute_inference("cifar_0-2_0.15", "backdoor_0-2_0.15_1-32.pickle", "Model__241226-1801__cifar_resnet32_E200_B128__backdoor_0-2_0.15_1-32__Cluster.pth", "Model__241223-1414__cifar_resnet32_E1_B128__backdoor_0-2_0.4_1-32__NoCluster.pth")
