import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.dataset import Subset

from defense.boost import filter_noise

from evaluation import train
from evaluation import runoff

LOGGER = None

def create_folder(dataset, model):
    from datetime import datetime
    test_name = dataset.split("/")[1].replace(".pickle", "")
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    ## Base folder for current test
    folder_path = os.path.join(f"results/{test_name}_{model}_{timestamp}")
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
        scheduler_constructor=None, filtering=True):
    
    test_folder, model_folder = create_folder(dataset, try_get_list(model_constructor, 1).__name__)
    global LOGGER
    LOGGER = open(os.path.join(test_folder, "data.txt"), "w")

    LOGGER.write(f"Dataset: {dataset}")

    print("Starting image loading")
    clean_testset, clean_testloader = dataset_loader("clean", batch_size, train=False)
    poison_testset, poison_testloader = dataset_loader(dataset, batch_size, train=False)
    poison_trainset, poison_trainloader = dataset_loader(dataset, batch_size, train=True)
    print("Finish image loading")

    if filtering:
        print("Starting poison filtering")
        ## Filtering parameters
        m_ctr, op_ctr, s_ctr = (try_get_list(constructor, 0) for constructor in [model_constructor, optimizer_constructor, scheduler_constructor])
        alpha, beta = 8, 4
        data_perc = .96 / alpha
        ground_truth_clean = np.array([i in poison_trainset.clean_samples for i in range(len(poison_trainset.targets))])

        clean, net = filter_noise(m_ctr,
                        batch_size,
                        poison_trainset, len(poison_trainset.classes), poison_trainset.true_targets,
                        op_ctr, scheduler_fn=s_ctr,
                        data_perc=data_perc,
                        boost=alpha,
                        beta=beta,
                        ground_truth_clean=ground_truth_clean,
                        device=device)
        
        true_clean = np.zeros(len(poison_trainset))
        true_clean[poison_trainset.clean_samples] = 1
        false_pos, false_neg, true_pos, true_neg = compute_poison_stats(clean, true_clean)

        LOGGER.write("\nResults of identification of poisoned images:")
        LOGGER.write(f"\n\t Poisoned removed images (detected as poison): {true_pos}")
        LOGGER.write(f"\n\t Poisoned non-removed images (detected as clean): {false_neg}")
        LOGGER.write(f"\n\t Clean non-removed images (detected as clean): {true_neg}")
        LOGGER.write(f"\n\t Clean removed images (detected as poison): {false_pos}")

        print("Finished poison filtering")


        cleanset = Subset(poison_trainset, [i for i in range(len(poison_trainset)) if clean[i]])
    else: 
        cleanset = poison_trainset

    LOGGER.close()
    trainloader = torch.utils.data.DataLoader(cleanset, batch_size=batch_size, shuffle=True, num_workers=2)

    
    m_ctr, op_ctr, s_ctr = (try_get_list(constructor, 1) for constructor in [model_constructor, optimizer_constructor, scheduler_constructor])
    net = m_ctr().to(device)

    optimizer = op_ctr(net.parameters())
    scheduler = s_ctr(optimizer) if s_ctr is not None else None

    ## Training
    with open(os.path.join(test_folder, "train.txt"), "w") as LOGGER:
        print("Starting model training")
        LOGGER.write(f"\n Starting model training:")
        if m_ctr.__name__ == "Net":
            runoff.log = LOGGER
            runoff.train(net, epochs, trainloader, trainloader, model_folder)
        else:
            criterion = torch.nn.CrossEntropyLoss()
            train.train(net, criterion, optimizer, epochs, trainloader, device, 0, scheduler=scheduler, log=LOGGER)
            torch.save(net.state_dict(), os.path.join(model_folder, f"BestModel_Epoch{epochs}"))

    ## Load best model from training
    model_file = os.path.join(model_folder, os.listdir(model_folder)[0])
    net = m_ctr().to(device)
    net.load_state_dict(torch.load(model_file))

    with open(os.path.join(test_folder, "test.txt"), "w") as LOGGER:
        print("Starting testing")
        if m_ctr.__name__ == "Net":
            runoff.log = LOGGER
            LOGGER.write(f"\n Testing for clean testser:")
            runoff.test(net, clean_testloader)
            LOGGER.write(f"\n Testing for poisoned testser:")
            runoff.test(net, poison_testloader)
        else:
            source, target = poison_trainset.source, poison_trainset.target

            clean_accuracy, clean_misclassification = train.test(net, clean_testloader, device, source=source)
            LOGGER.write(f"\nAccuracy on clean testset: {clean_accuracy}")

            poison_accuracy, poison_misclassification = train.test(net, poison_testloader, device, source=source, target=target)
            p = sum(poison_misclassification) / len(poison_misclassification)
            LOGGER.write(f"\nAccuracy on poison testset: {poison_accuracy}")
            LOGGER.write(f"\nPoison misclassification: {p}")

            poison_misclassification = [p and c for p, c in zip(poison_misclassification, clean_misclassification)]
            p = sum(poison_misclassification) / len(poison_misclassification)
            LOGGER.write(f"\nTargeted misclassification: {p}")




