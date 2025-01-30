docker run -v ./database:/home/app/database --network host executer:latest --dataset cifar --poison database/backdoor/backdoor_0-2_0.2_1-32.pickle --model resnet32 --epochs 200 --batch 128
