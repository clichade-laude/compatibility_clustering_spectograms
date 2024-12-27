import argparse, os
from datetime import datetime

import poisoning.poison, clustering.cluster, training.train, testing.test

def log_time():
    return datetime.now().strftime("[%d/%m %H:%M]")

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
    if args.poison:
        print(f"{log_time()} Poisoning | Executing")
        args.dataset = poisoning.poison.poison(args.dataset, args.poison, args.folder)
        if args.cluster:
            print(f"{log_time()} Clustering | Executing")
            clustering.cluster.cluster(args.dataset, args.model, args.batch, args.folder)
    print(f"{log_time()} Training | Executing")
    training.train.execute_training(args.dataset, args.model, args.epochs, args.batch, args.folder, args.cluster)
    print(f"{log_time()} Testing | Executing")
    testing.test.execute_testing(args.dataset, args.orig_dataset, args.batch, args.folder, args.poison)
    print(f"{log_time()} Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str, help='Name of the dataset to poison')
    parser.add_argument("--poison", "-p", type=str, help='Path to the pickle file with the poison info')
    parser.add_argument("--model", "-m", type=str, help='CNN model to perform clustering', choices=["resnet32", "resnet18"], default="resnet32")
    parser.add_argument("--epochs", "-e", default=200, type=int, help='Number of epochs to train the model')
    parser.add_argument("--batch", "-b", default=128, type=int, help='Batch size to execute training and testing')
    parser.add_argument("--cluster", action="store_false", help="Indicates whether to load cleaned samples")
    args = parser.parse_args()
    args.folder = create_folder(args)
    args.orig_dataset = args.dataset
    print(f"{log_time()} Saving results on {args.folder}", flush=True)
    main(args)