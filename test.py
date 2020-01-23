import torchvision

resnet = torchvision.models.resnet50()

print(list(resnet.children()))