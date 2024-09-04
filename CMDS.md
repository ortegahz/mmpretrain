# data

openxlab dataset get --dataset-repo OpenDataLab/ImageNet-1K

# train

bash ./tools/dist_train.sh configs/resnet/resnet18_8xb16_cifar10.py 8

# infer

python demo/image_demo.py demo/airplane1.png configs/resnet/resnet18_8xb16_cifar10.py --checkpoint work_dirs/resnet18_8xb16_cifar10/epoch_200.pth --device cuda:0