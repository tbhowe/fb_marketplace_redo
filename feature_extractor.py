#%%
import torch
from classifier import MinimalResNet, FeatureExtractor
from dataset import ImagesDataset
from torchvision import transforms
size = 64
transform = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomCrop((size,size), pad_if_needed=True),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomHorizontalFlip(p=0.25),
    # transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


dataset=ImagesDataset(transform=transform)
model=FeatureExtractor()
print(model.layers)

# %%
