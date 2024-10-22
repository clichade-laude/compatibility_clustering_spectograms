import pickle, os, shutil, argparse
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

    poisoned_path = create_posioned_db(dataset_name, params['fraction_poisoned'])
    logger = open(os.path.join(poisoned_path, "poison_info.txt"), "w")
    logger.write(f"Dataset: {dataset_name}")

    clean_imgs = move_clean_imgs(poisoned_path, dataset_path, dataset_classes, source_class, logger)
    
    source_images = np.array(sorted(os.listdir(os.path.join(dataset_path, source_class))))
    poison_count = int(params['fraction_poisoned'] * source_images.shape[0])
    clean_count = source_images.shape[0] - poison_count
    poisoned_imgs = np.random.choice(source_images, size=poison_count, replace=False)
    np.savez(os.path.join(poisoned_path, 'poison_info.npz'), **{source_class: poisoned_imgs})

    logger.write("\nPoisoned Class Info:")
    logger.write(f"\n\tClass: {source_class}")
    logger.write(f"\n\tTarget class: {target_class}")
    logger.write(f"\n\tPoisoned percentage: {params['fraction_poisoned']}")
    logger.write(f"\n\tPoisoned images: {poison_count}")
    logger.write(f"\n\tClean images: {clean_count}")
    logger.write(f"\nTotal Clean images: {clean_imgs + clean_count}")
    logger.close()

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

def create_posioned_db(dataset_name, fraction_poisoned):
    poisoned_path = os.path.join('database/poisoned', f"{dataset_name}-{fraction_poisoned}")
    ## Create poisoned folder, deleting it if previously existed
    if os.path.exists(poisoned_path):
        shutil.rmtree(poisoned_path)
    os.makedirs(poisoned_path)
    return poisoned_path

def move_clean_imgs(poisoned_path, dataset_path, dataset_classes, source_class, logger):
    logger.write("\nClean Classes Images:")
    clean_imgs = 0
    ## Copy all the classes from original to the poisoned database
    for ds_css in dataset_classes:
        if ds_css == source_class:
            ## If source class, create only the folder
            os.makedirs(os.path.join(poisoned_path, source_class))
        else:
            ## Copy images from original path to the poisoned one
            shutil.copytree(os.path.join(dataset_path, ds_css), os.path.join(poisoned_path, ds_css))

            ## Logger information
            css_imgs = len(os.listdir(os.path.join(poisoned_path, ds_css)))
            logger.write(f"\n\t{ds_css}: {css_imgs}")
            clean_imgs += css_imgs
    return clean_imgs


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str, help='Name of the dataset to poison')
    parser.add_argument("--poison", "-p", required=True, type=str, help='Path to the pickle file with the poison info')
    args = parser.parse_args()
    print(args.dataset, args.poison)
    poison(args.dataset, args.poison)