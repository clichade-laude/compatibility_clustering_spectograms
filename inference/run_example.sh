docker run --name inference_container \
    -v ./database:/home/app/database -v compatibility_clustering_spectograms_prometheus-targets:/etc/prometheus/targets \
    --network compatibility_clustering_spectograms_my_network inference:latest \
    --dataset cifar_0-2_0.15 --poison backdoor_0-2_0.4_1-32.pickle \
    --model Model__241227-1948__cifar_resnet32_E200_B128__backdoor_0-2_0.4_1-32__Cluster.pth \
            Model__250107-0652__cifar_resnet32_E200_B128__backdoor_0-2_0.4_1-32__NoCluster.pth \
            Model__250102-1307__cifar_resnet32_E200_B128__NoPoison__NoCluster.pth