import argparse, os

import poisoning.poison, clustering.cluster, training.train, testing.test

def main(dataset, poison_params, model_name, epochs, batch_size, doPoison: bool, doCluster: bool):
    params_name = poison_params.split('/')[-1].split('backdoor')[1].split(".pickle")[0]
    test_path = f"results/{dataset}{params_name}_{model_name}_Eph{epochs}"
    os.makedirs(test_path)
    if doPoison:
        ds_path = poisoning.poison.poison(dataset, poison_params)
        os.rename(os.path.join(ds_path, "poison_info.txt"), os.path.join(test_path, "poison_info.txt"))
        poisoned_ds_name = ds_path.replace('dataset/poisoned/', '')
        if doCluster:
            clustering.cluster.cluster(poisoned_ds_name, model_name, batch_size)
            os.rename(os.path.join(ds_path, "clustering.txt"), os.path.join(test_path, "clustering.txt"))
    else: 
        ds_path = os.path.join('database/original', dataset, 'train')
    output_name = training.train.execute_training(ds_path, model_name, epochs, batch_size, doCluster)
    output_path = os.path.join(test_path, output_name)
    os.rename(os.path.join('database/models', output_name + '.txt'), output_path + '.txt')
    os.rename(os.path.join('database/models', output_name + '.pth'), output_path + '.pth')
    testing.test.execute_testing(dataset, output_path + '.pth', batch_size)
    os.rename(os.path.join('database/models', output_name + '_model.txt'), output_path + '_model.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str, help='Name of the dataset to poison')
    parser.add_argument("--poison_info", "-p", required=True, type=str, help='Path to the pickle file with the poison info')
    parser.add_argument("--model", "-m", type=str, help='CNN model to perform clustering', choices=["resnet32", "resnet18"], default="resnet32")
    parser.add_argument("--epochs", "-e", default=200, type=int, help='Number of epochs to train the model')
    parser.add_argument("--batch", "-b", default=128, type=int, help='Batch size to execute training and testing')
    parser.add_argument("--poison", action="store_false", help="Indicates whether to poison the dataset")
    parser.add_argument("--cluster", action="store_false", help="Indicates whether to load cleaned samples")
    args = parser.parse_args()
    print(args.dataset, args.poison_info, args.model, args.epochs, args.batch, args.poison, args.cluster)
    main(args.dataset, args.poison_info, args.model, args.epochs, args.batch, args.poison, args.cluster)