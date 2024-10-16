import sys
sys.path.append('/home/laude/compatibility_clustering_spectograms/')

import os
import torch
import numpy as np

from utils.models import get_model_info
from utils.dataset import load_data
from clustering.boost import filter_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_poison_stats(keep, clean):
    false_pos = sum(np.logical_and(np.logical_not(keep), clean))
    false_neg = sum(np.logical_and(keep, np.logical_not(clean)))
    true_pos = sum(np.logical_and(np.logical_not(keep), np.logical_not(clean)))
    true_neg = sum(np.logical_and(keep, clean))
    return false_pos, false_neg, true_pos, true_neg

def cluster(dataset_name, batch_size, model_name):
    dataset_path = os.path.join("database/poisoned", dataset_name)
    ## Load dataset and dataloader
    dataset, _ = load_data(dataset_path, batch_size)
    ## Retrieve model and parameteres according to the selected one and the operation type
    model, optim, sched = get_model_info(model_name, operation="filter")
    ## Clustering parameters
    # alpha, beta = 8, 4
    alpha, beta = 1, 1
    data_perc = .96 / alpha

    clean, net = filter_noise(
                    model, batch_size,
                    dataset, len(dataset.classes), None,
                    optim, scheduler_fn=sched,
                    data_perc=data_perc, boost=alpha, beta=beta, device=device,
                    bag=1)
                    ## Los campos clean_labels (None) y ground_truth_clean (omitido) no se han añadido
                    ## ya que en teoría ni deberíamos poder conocerlos ni deberían hacer falta.

    # clean = np.random.choice([np.True_, np.False_], size=len(dataset))
    dataset.export_clean(clean)
    ## Do not retrieve poisoned information until finishing the clustering
    dataset.obtain_poisoned()
    true_clean = np.zeros(len(dataset))
    true_clean[dataset.clean_samples] = 1
    false_pos, false_neg, true_pos, true_neg = compute_poison_stats(clean, true_clean)

    print("\nResults of identification of poisoned images:")
    print(f"\n\t Poisoned removed images (detected as poison): {true_pos}")
    print(f"\n\t Poisoned non-removed images (detected as clean): {false_neg}")
    print(f"\n\t Clean non-removed images (detected as clean): {true_neg}")
    print(f"\n\t Clean removed images (detected as poison): {false_pos}")
    print()
   

cluster('cifar', 256, 'resnet32')
