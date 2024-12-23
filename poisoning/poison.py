import pickle, os, shutil, argparse
import numpy as np
from os.path import join

from utils.mqtt import connect_node, publish_mqtt, log_time
from PIL import Image

np.random.seed(42)

def poison(dataset_name, poison_params, test_folder):
    ## Obtain original database paths
    dataset_path = join('database/original', dataset_name, 'train')

    ## Load poison parameteres and obtain classes and names
    with open(f'{poison_params}', 'rb') as f:
        params = pickle.load(f)
    
    dataset_classes = sorted(os.listdir(dataset_path))
    source_class = dataset_classes[params['source']]
    target_class = dataset_classes[params['target']]

    ## Obtain test path and create logger
    test_path = join("database", "results", test_folder)
    logger = open(join(test_path, "poison_info.txt"), "w")
    logger.write(f"Dataset: {dataset_name}")

    ## Create poisoned entry in database for train and test
    poisoned_path  = create_posioned_db(dataset_name, params['source'], params['target'], params['fraction_poisoned'], "train")
    ## Move images from clean classes into new dataset
    clean_imgs = move_clean_imgs(poisoned_path, dataset_path, dataset_classes, source_class, logger)
    
    ## Set images to poison at trainset and save info
    source_images = np.array(sorted(os.listdir(join(dataset_path, source_class))))
    poison_count = int(params['fraction_poisoned'] * source_images.shape[0])
    clean_count = source_images.shape[0] - poison_count
    poisoned_imgs = np.random.choice(source_images, size=poison_count, replace=False)
    np.savez(join(poisoned_path, 'poison_info.npz'), **{source_class: poisoned_imgs})

    ## Log poisoned info
    logger.write("\nPoisoned Class Info:")
    logger.write(f"\n\tClass: {source_class}")
    logger.write(f"\n\tTarget class: {target_class}")
    logger.write(f"\n\tPoisoned percentage: {params['fraction_poisoned']}")
    logger.write(f"\n\tPoisoned images: {poison_count}")
    logger.write(f"\n\tClean images: {clean_count}")
    logger.write(f"\nTotal Clean images: {clean_imgs + clean_count}")
    logger.close()

    ## Poison and save train images
    for img_name in source_images:
        if img_name in poisoned_imgs:
            poison_image(dataset_path, poisoned_path, img_name, source_class, target_class, params)
        else:
            shutil.copyfile(join(dataset_path, source_class, img_name), join(poisoned_path, source_class, img_name))

    ## Poison and save test images
    poisoned_path  = create_posioned_db(dataset_name, params['source'], params['target'], params['fraction_poisoned'], "test")
    testset_path = join('database/original', dataset_name, 'test')
    move_clean_imgs(poisoned_path, testset_path, dataset_classes, source_class)
    for img_name in os.listdir(join(testset_path, source_class)):
        poison_image(testset_path, poisoned_path, img_name, source_class, target_class, params)
    np.savez(join(poisoned_path, 'poison_info.npz'), **{source_class: np.array(sorted(os.listdir(join(testset_path, source_class))))})
    
    return poisoned_path.split('/')[-2]


def create_posioned_db(dataset_name, source, target, fraction_poisoned, subset):
    poisoned_path = join('database/poisoned', f"{dataset_name}_{source}-{target}_{fraction_poisoned}", subset)
    ## Create poisoned folder, deleting it if previously existed
    if os.path.exists(poisoned_path):
        shutil.rmtree(poisoned_path)
    os.makedirs(poisoned_path)
    return poisoned_path

def move_clean_imgs(poisoned_path, dataset_path, dataset_classes, source_class, logger=None):
    logger.write("\nClean Classes Images:") if logger else None
    clean_imgs = 0
    ## Copy all the classes from original to the poisoned database
    for ds_css in dataset_classes:
        if ds_css == source_class:
            ## If source class, create only the folder
            os.makedirs(join(poisoned_path, source_class))
        else:
            ## Copy images from original path to the poisoned one
            shutil.copytree(join(dataset_path, ds_css), join(poisoned_path, ds_css))

            ## Logger information
            css_imgs = len(os.listdir(join(poisoned_path, ds_css)))
            logger.write(f"\n\t{ds_css}: {css_imgs}") if logger else None
            clean_imgs += css_imgs
    return clean_imgs

def poison_image(orig_path, goal_path, img_name, source_class, target_class, poison_params):
    orig_image = Image.open(join(orig_path, source_class, img_name))
    image = np.copy(np.asarray(orig_image))

    image[poison_params['position'][0]:poison_params['position'][0] + poison_params['size'], 
          poison_params['position'][1]:poison_params['position'][1] + poison_params['size'], :] = np.asarray(poison_params['color'])
    
    Image.fromarray(image).save(join(goal_path, target_class, img_name))

def on_message(client, userdata, msg):
    print(f"{log_time()} Node: poison | Executing", flush=True)
    import json
    params = json.loads(msg.payload)
    pois_dataset = poison(params['dataset'], params['poison'], params["folder"])

    publish_mqtt(client, "control", node=userdata, dataset=pois_dataset)

if __name__ == "__main__":
    connect_node("poison", "poison", on_message)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", "-d", required=True, type=str, help='Name of the dataset to poison')
    # parser.add_argument("--poison", "-p", required=True, type=str, help='Path to the pickle file with the poison info')
    # args = parser.parse_args()
    # print(args.dataset, args.poison)
    # poison(args.dataset, args.poison)
    # poison("cifar", "database/backdoor/backdoor_0-2_0.5_1-32.pickle", "xxx")