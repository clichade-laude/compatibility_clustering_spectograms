import sys
sys.path.append("/home/laude/compatibility_clustering_spectograms/")
import torch
from evaluation.run_defense import run_defense
from data.cifar10 import cifar10_loader

from model.preact_resnet import PreActResNet18
from model.resnet_paper import resnet32
from model.modelnet import Net

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_datasets():
    pairs = [(0, 1)]
    poison_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.]
    ds = []
    for poison in poison_levels:
        for source, target in pairs:
            ds.append(f"datasets/cifar-backdoor-{source}-to-{target}-{poison}.pickle")
    return ds
    
def run(dataset, model_filter, model_train, filtering):
    if all(x not in model_filter for x in ["32", "18"]):
        assert False, f"model not parseable from {model_filter}"
    if all(x not in model_train for x in ["32", "custom"]):
        assert False, f"model not parseable from {model_train}"

    model_ctor = resnet32 if "32" in model_filter else PreActResNet18
    model_ctrain = resnet32 if "32" in model_train else Net

    if model_ctor == resnet32:
        train_op_ctr = lambda parameters: torch.optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=1e-4)
        train_s_ctr = lambda optimizer, e=200: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        filter_op_ctr = train_op_ctr
        filter_s_ctr = lambda optimizer, e=100: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90], gamma=0.1)
    elif model_ctor == PreActResNet18:
        train_op_ctr = lambda parameters: torch.optim.SGD(parameters, lr=0.02, momentum=0.9, weight_decay=5e-4)
        train_s_ctr = lambda optimizer, e=200: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=0.1)
        filter_op_ctr = train_op_ctr
        filter_s_ctr = train_s_ctr

    testsets = ['clean', dataset]
    run_defense(
        dataset,
        [model_ctor, model_ctrain], 
        [filter_op_ctr, train_op_ctr], cifar10_loader, 
        500, 512, 
        device, [filter_s_ctr, train_s_ctr], filtering)

if __name__ == "__main__":
    for dataset in get_datasets():
        run(dataset, "resnet32", "resnet32", True) 
        run(dataset, "resnet32", "resnet32", False) 

