from mmpretrain import ImageClassificationInferencer

image = 'demo/airplane1.png'
config = 'configs/resnet/resnet18_8xb16_cifar10.py'
checkpoint = 'work_dirs/resnet18_8xb16_cifar10/epoch_200.pth'
inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device='cuda')
result = inferencer(image)[0]
print(result['pred_class'])
