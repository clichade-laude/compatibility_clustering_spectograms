import sys, os
import torch

from evaluation import runoff, run_defense
from model.modelnet import Net
from data.cifar10 import cifar10_loader

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def test(exp_path, batch_size):
    global LOGGER
    LOGGER = open(os.path.join(exp_path, "test.txt"), "w") 
    run_defense.LOGGER = LOGGER

    ## Load testsets using the poison file
    poison_file = os.path.join("datasets", exp_path.split("_")[0].split("/")[1] + ".pickle")
    print("Starting image loading")
    clean_testset, clean_testloader = cifar10_loader("clean", batch_size, train=False)
    poison_testset, poison_testloader = cifar10_loader(poison_file, batch_size, train=False)
    print("Finish image loading")

    ## Load saved model to perform the tests
    model_path = os.path.join(exp_path, "models")
    model_file = os.path.join(model_path, os.listdir(model_path)[0])
    model = Net().to(device)
    model.load_state_dict(torch.load(model_file))
    
    ## Execute testing
    runoff.log = LOGGER
    LOGGER.write(f"\n Testing for clean testser:")
    runoff.test(model, clean_testloader)
    LOGGER.write(f"\n Testing for poisoned testser:")
    runoff.test(model, poison_testloader)   
    
if __name__ == "__main__":
    exp_path = sys.argv[1]
    test(exp_path, 512)
