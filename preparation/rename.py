import os

"""
    Recursively loops the CIFAR folders and add the class name as part of the image name
"""
def loop_folder(name, path):
    folder_items = os.listdir(path)
    if ".png" in folder_items[0]:
        for file in folder_items:
            os.rename(os.path.join(path, file), os.path.join(path, f"{name}-{file}"))
    else:
        for dir in folder_items:
            loop_folder(dir, os.path.join(path, dir))


if __name__ == "__main__":
    cifar_path = "database/original/cifar"
    loop_folder(None, cifar_path)