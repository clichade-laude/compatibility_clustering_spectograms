import torch

from model.preact_resnet import PreActResNet18
from model.resnet_paper import resnet32
from model.modelnet import Net as CustomNet

model_info = {
        "resnet32": {
            "model": resnet32,
            "optim": lambda parameters: torch.optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=1e-4),
            "sched": {
                "filter": lambda optimizer, e=100: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90], gamma=0.1),
                "train": lambda optimizer, e=200: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
                }
        },
        "resnet18": {
            "model": PreActResNet18,
            "optim": lambda parameters: torch.optim.SGD(parameters, lr=0.02, momentum=0.9, weight_decay=5e-4),
            "sched": lambda optimizer, e=200: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=0.1)
        },
        "customnet": {
            "model": CustomNet,
            "optim": lambda parameters: torch.optim.Adam(parameters, lr=0.0005),
            "sched": None
        }
    }

def get_op_info(optim_dict, operation):
    return optim_dict[operation] if isinstance(optim_dict, dict) else optim_dict

def get_model_info(model_name, operation):
    model = model_info[model_name]['model']
    optim = get_op_info(model_info[model_name]['optim'], operation)
    sched = get_op_info(model_info[model_name]['sched'], operation)

    return model, optim, sched

