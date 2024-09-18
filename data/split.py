import os, numpy as np

np.random.seed(42)
ds_folder = "datasets"
classes = ["jammer", "clean"]

def loop_folder(dir, ext):
    return [file for file in os.listdir(dir) if ext in file]

## Given the length of a dataset, return indices for train, valid and test
def create_split_samplers(ds_len, test_rate):
    ## Create and shuffle lists of indices, from where we will extract the indices per split
    ds_idxs = np.arange(ds_len)
    np.random.shuffle(ds_idxs)
    ## Create the indices to stopping the sampler per split
    test_split = int(np.floor(test_rate * ds_len))
    ## Return samplers per split
    return ds_idxs[:test_split], ds_idxs[test_split:]

def main(ds_name):
    ds_path = os.path.join(ds_folder, ds_name)
    images_css = {type_css: loop_folder(os.path.join(ds_path, type_css), ".png") for type_css in classes}

    for type_css in classes:
        splits_idxs = create_split_samplers(len(images_css[type_css]), 0.1)
        for i, split in enumerate(["test", "train"]):
            goal_path = os.path.join(ds_path, split, type_css)
            os.makedirs(goal_path, exist_ok=True)
            for idx in splits_idxs[i]:
                os.rename(os.path.join(ds_path, type_css, images_css[type_css][idx]), os.path.join(goal_path, images_css[type_css][idx]))

main("spectrogram-dataset")