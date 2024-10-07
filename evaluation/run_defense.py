import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.dataset import Subset

from defense.boost import filter_noise

from model.preact_resnet import PreActResNet18
from model.resnet_paper import resnet32

from model.modelnet import Net
from evaluation import runoff

LOGGER = None

def create_folder(dataset):
    from datetime import datetime
    test_name = dataset.split("/")[1].replace(".pickle", "")
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    ## Base folder for current test
    folder_path = os.path.join(f"results/{test_name}_{timestamp}")
    # Models folder for current test
    model_path = os.path.join(folder_path, "models")
    ## Create folders
    os.makedirs(model_path)

    return folder_path, model_path

def try_get_list(maybe_list, idx):
    if isinstance(maybe_list, list):
        return maybe_list[idx]
    else:
        return maybe_list

def compute_poison_stats(keep, clean):
    false_pos = sum(np.logical_and(np.logical_not(keep), clean))
    false_neg = sum(np.logical_and(keep, np.logical_not(clean)))
    true_pos = sum(np.logical_and(np.logical_not(keep), np.logical_not(clean)))
    true_neg = sum(np.logical_and(keep, clean))
    return false_pos, false_neg, true_pos, true_neg

def run_defense(dataset,
        model_constructor, optimizer_constructor, 
        dataset_loader, epochs, batch_size, device, 
        scheduler_constructor=None):
    
    test_folder, model_folder = create_folder(dataset)
    global LOGGER
    LOGGER = open(os.path.join(test_folder, "data.txt"), "w")

    LOGGER.write(f"Dataset: {dataset}")

    print("Starting image loading")
    clean_testset, clean_testloader = dataset_loader("clean", batch_size, train=False)
    poison_testset, poison_testloader = dataset_loader(dataset, batch_size, train=False)
    poison_trainset, poison_trainloader = dataset_loader(dataset, batch_size, train=True)
    print("Finish image loading")

    m_ctr = try_get_list(model_constructor, 0)
    op_ctr = try_get_list(optimizer_constructor, 0)
    s_ctr = try_get_list(scheduler_constructor, 0)

    alpha = 8
    data_perc = .96 / alpha
    beta = 4
    ground_truth_clean = np.array([i in poison_trainset.clean_samples for i in range(len(poison_trainset.targets))])

    print("Starting poison filtering")

    clean, net = \
            filter_noise(m_ctr,
                         batch_size,
                         poison_trainset, 10, poison_trainset.true_targets,
                         op_ctr, scheduler_fn=s_ctr,
                         data_perc=data_perc,
                         boost=alpha,
                         beta=beta,
                         ground_truth_clean=ground_truth_clean,
                         device=device)
    
    print("Finished poison filtering")

    true_clean = np.zeros(len(poison_trainset))
    true_clean[poison_trainset.clean_samples] = 1
    false_pos, false_neg, true_pos, true_neg = compute_poison_stats(
        clean, true_clean)

    LOGGER.write("\nResults of identification of poisoned images:")
    LOGGER.write(f"\n\t Poisoned removed images (detected as poison): {true_pos}")
    LOGGER.write(f"\n\t Poisoned non-removed images (detected as clean): {false_neg}")
    LOGGER.write(f"\n\t Clean non-removed images (detected as clean): {true_neg}")
    LOGGER.write(f"\n\t Clean removed images (detected as poison): {false_pos}")
    LOGGER.close()


    cleanset = Subset(poison_trainset, [i for i in range(len(poison_trainset)) if clean[i]])
    # cleanset = Subset(poison_trainset, [i for i in range(len(poison_trainset))])
    # cleanset = torch.load("cleanset.pth")

    trainloader = torch.utils.data.DataLoader(
            cleanset, batch_size=batch_size, shuffle=True, num_workers=2)

    # torch.save(trainloader, "cleanloader.pth")
    # torch.save(poison_trainloader, "trainloader.pth")
    # torch.save(clean_testloader, "testloader_clean.pth")
    # torch.save(poison_testloader, "testloader_poison.pth")

    ## Training
    with open(os.path.join(test_folder, "train.txt"), "w") as LOGGER:
        LOGGER.write(f"\n Starting model training:")
        runoff.log = LOGGER
        model = Net().to(device)
        runoff.train(model, epochs, trainloader, trainloader, model_folder)

    ## Load best model from training
    model_file = os.path.join(model_folder, os.listdir(model_folder)[0])
    model = Net().to(device)
    model.load_state_dict(torch.load(model_file))

    ## Perform testing
    with open(os.path.join(test_folder, "test.txt"), "w") as LOGGER:
        runoff.log = LOGGER
        LOGGER.write(f"\n Testing for clean testser:")
        runoff.test(model, clean_testloader)
        LOGGER.write(f"\n Testing for poisoned testser:")
        runoff.test(model, poison_testloader)



