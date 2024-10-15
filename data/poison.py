import pickle
import numpy as np
from torchvision import datasets

class PoisonDataset(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, poison_params=None):
        super(PoisonDataset, self).__init__(root, transform, target_transform, download=download)
        self.train = train
        self.clean_samples = None 
        self.method = None
        self.position = None
        self.color = None
        self.fraction_poisoned = None
        self.poison_seed = None
        self.source = None
        self.target = None
        self.true_targets = np.array(self.targets)
        self.targets = np.array(self.targets)

        self.get_info(poison_params)

        if poison_params is not None:    
            with open(f'{poison_params}', 'rb') as f:
                params = pickle.load(f)

            self.method = params['method']
            self.position = params['position']
            self.color = params['color']
            self.fraction_poisoned = float(params['fraction_poisoned'])
            self.poison_seed = params['seed']
            self.source = params['source']
            self.target = params['target']

            self.poison()
    
    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

    def get_info(self, poison_params):
        from evaluation.run_defense import LOGGER
        LOGGER.write("\nSubset: train") if "train" in self.root else LOGGER.write("\nSubset: test")
        LOGGER.write(f"\n\t Total samples: {self.targets.size}")
        for cls_name, cls_idx in self.class_to_idx.items():
            LOGGER.write(f"\n\t\t {cls_name} samples: {np.sum(self.targets == cls_idx)}")
        LOGGER.write("\n\t Clean.") if not poison_params else LOGGER.write("\n\t Poisoned:")


    def poison(self):
        method = self.method
        position = self.position
        color = self.color
        fraction_poisoned = self.fraction_poisoned
        seed = self.poison_seed
        source = self.source
        target = self.target

        class_idxs = np.where(np.array(self.targets) == source)[0]
        poison_count = int(fraction_poisoned * class_idxs.shape[0])
        if self.train:
            st = np.random.get_state()
            np.random.seed(seed)
            poisoned_idxs = \
                    np.random.choice(class_idxs, size=poison_count, replace=False)
            np.random.set_state(st)
        else:
            poisoned_idxs = class_idxs

        for i in poisoned_idxs:
            image = self.data[i]
            poisoned_image = poison_image(image, method, position, color)
            self.data[i] = poisoned_image
            self.targets[i] = target

        poisoned_images = np.isin(np.arange(len(self.targets)), poisoned_idxs)
        self.clean_samples = np.where(poisoned_images == 0)[0]

        from evaluation.run_defense import LOGGER
        LOGGER.write(f"\n\t\t Poisoned class: {self.classes[source]}")
        LOGGER.write(f"\n\t\t Poisoned percentage: {self.fraction_poisoned}")
        LOGGER.write(f"\n\t\t Total/Class poisoned images: {poisoned_idxs.size}")
        LOGGER.write(f"\n\t\t Class clean images: {class_idxs.size-poisoned_idxs.size}")
        LOGGER.write(f"\n\t\t Total clean images: {len(self.targets) - poisoned_idxs.size}")


def poison_image(image, method, position, color):
    """
    adapted from https://github.com/MadryLab/backdoor_data_poisoning/blob/master/dataset_input.py
    method = "pixel" or "pattern" or "ell"
    """
    poisoned = np.copy(image)
    col_arr = np.asarray(color)

    if method == 'pixel':
        poisoned[position[0], position[1], :] = col_arr
    elif method == 'pattern':
        poisoned[position[0], position[1], :] = col_arr
        poisoned[position[0] + 1, position[1] + 1, :] = col_arr
        poisoned[position[0] - 1, position[1] + 1, :] = col_arr
        poisoned[position[0] + 1, position[1] - 1, :] = col_arr
        poisoned[position[0] - 1, position[1] - 1, :] = col_arr
    elif method == 'ell':
        poisoned[position[0], position[1], :] = col_arr
        poisoned[position[0] + 1, position[1], :] = col_arr
        poisoned[position[0], position[1] + 1, :] = col_arr
    return poisoned

