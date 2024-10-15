import pickle, os, shutil
import numpy as np

from PIL import Image

np.random.seed(42)

def poison(dataset_name, poison_params):
    dataset_path = os.path.join('database/original', dataset_name, 'train')
    with open(f'{poison_params}', 'rb') as f:
        params = pickle.load(f)
    
    dataset_classes = sorted(os.listdir(dataset_path))
    source_class = dataset_classes[params['source']]
    target_class = dataset_classes[params['target']]

    source_images = np.array(sorted(os.listdir(os.path.join(dataset_path, source_class))))

    poisoned_path = create_poisoned_db(dataset_name, dataset_path, dataset_classes, source_class)
    poison_count = int(params['fraction_poisoned'] * source_images.shape[0])
    poisoned_imgs = np.random.choice(source_images, size=poison_count, replace=False)
    np.savez(os.path.join(poisoned_path, 'poison_info.npz'), **{source_class: poisoned_imgs})

    for img_name in source_images:
        if img_name in poisoned_imgs:
            ## Load image and transform to array
            image = Image.open(os.path.join(dataset_path, source_class, img_name))
            image = np.asarray(image)
            ## Poison it and return to image
            poisoned_image = poison_image(image, params['method'], params['position'], params['color'])
            poisoned_image = Image.fromarray(poisoned_image)
            ## Save it on target class
            poisoned_image.save(os.path.join(poisoned_path, target_class, img_name))
        else:
            shutil.copyfile(os.path.join(dataset_path, source_class, img_name), os.path.join(poisoned_path, source_class, img_name))


def create_poisoned_db(dataset_name, dataset_path, dataset_classes, source_class):
    ## Create poisoned folder, deleting it if previously existed
    poisoned_path = os.path.join('database/poisoned', dataset_name)
    if os.path.exists(poisoned_path):
        shutil.rmtree(poisoned_path)
    os.makedirs(poisoned_path)
    
    ## Copy all the classes from original to the poisoned database
    for ds_css in dataset_classes:
        if ds_css == source_class:
            ## If source class, create only the folder
            os.makedirs(os.path.join(poisoned_path, source_class))
        else:
            shutil.copytree(os.path.join(dataset_path, ds_css), os.path.join(poisoned_path, ds_css))

    return poisoned_path

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

if __name__ == "__main__":
    poison('cifar', 'datasets/cifar-backdoor-0-to-2-0.5.pickle')