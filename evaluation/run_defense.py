import os
import numpy as np
import torch
from torch.utils.data.dataset import Subset

from evaluation import train
from defense.boost import filter_noise

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

    source = poison_trainset.source
    target = poison_trainset.target

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
                         poison_trainset, len(poison_trainset.classes), poison_trainset.true_targets,
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
    trainloader = torch.utils.data.DataLoader(cleanset, batch_size=batch_size, shuffle=True, num_workers=2)


    net = m_ctr().to(device)
    optimizer = op_ctr(net.parameters())
    if s_ctr is not None:
        scheduler = s_ctr(optimizer)
    else:
        scheduler = None

    with open(os.path.join(test_folder, "train.txt"), "w") as LOGGER:
        LOGGER.write(f"\n Starting model training:")
        train.log = LOGGER
        criterion = torch.nn.CrossEntropyLoss()
        train.train(net, criterion, optimizer, epochs, trainloader, device, 0, scheduler=scheduler, log=LOGGER)

    ## Perform testing
    with open(os.path.join(test_folder, "test.txt"), "w") as LOGGER:
        clean_accuracy, clean_misclassification = train.test(net, clean_testloader, device, source=source)
        LOGGER.write(f"\nAccuracy on clean testset: {clean_accuracy}")

        poison_accuracy, poison_misclassification = train.test(net, poison_testloader, device, source=source, target=target)
        p = sum(poison_misclassification) / len(poison_misclassification)
        LOGGER.write(f"\nAccuracy on poison testset: {poison_accuracy}")
        LOGGER.write(f"\nPoison misclassification: {p}")

        poison_misclassification = [p and c for p, c in zip(poison_misclassification, clean_misclassification)]
        p = sum(poison_misclassification) / len(poison_misclassification)
        LOGGER.write(f"\nTargeted misclassification: {p}")


