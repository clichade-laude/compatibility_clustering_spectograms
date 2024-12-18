import argparse, os
from datetime import datetime

from utils.mqtt import connect_mqtt, publish_mqtt

def create_folder(args):
    ## Format variables for the name
    poison = "NoPoison" if not args.poison else args.poison.split("/")[-1].replace(".pickle","")
    cluster = "Cluster" if args.cluster and args.poison else "NoCluster"
    date = datetime.now().strftime("%y%m%d-%H%M")
    ## Create name and folder
    folder_name = f"{date}__{args.dataset}_{args.model}_E{args.epochs}_B{args.batch}__{poison}__{cluster}"
    os.makedirs(f"database/results/{folder_name}")
    return folder_name

def main(args):
    args.folder = create_folder(args)
    args = args.__dict__
    args["node"] = "start"
    client = connect_mqtt("start", None)
    publish_mqtt(client, "control", **args)
    client.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str, help='Name of the dataset to poison')
    parser.add_argument("--poison", "-p", type=str, help='Path to the pickle file with the poison info')
    parser.add_argument("--model", "-m", type=str, help='CNN model to perform clustering', choices=["resnet32", "resnet18"], default="resnet32")
    parser.add_argument("--epochs", "-e", default=200, type=int, help='Number of epochs to train the model')
    parser.add_argument("--batch", "-b", default=128, type=int, help='Batch size to execute training and testing')
    parser.add_argument("--cluster", action="store_false", help="Indicates whether to load cleaned samples")
    main(parser.parse_args())