from pytorch_unet import UNet
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import CameDataset
from torch.utils.data import DataLoader
import torch.optim as optim

from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
from tqdm import tqdm
import torch.nn as nn
import poptorch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
import math
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class ModelwithLoss(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x, labels=None):
        output = self.model(x)
        if labels is not None:
            loss = self.criterion(output, labels)
            return output, poptorch.identity_loss(loss, reduction='mean')
        return output

def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    return loss

num_epochs = 40
num_class = 1

model = UNet(num_class)

opts = poptorch.Options()
opts.replicationFactor(1)
opts.Training.gradientAccumulation(13)
opts.TensorLocations.setOptimizerLocation(
   poptorch.TensorLocationSettings().useOnChipStorage(False))
opts.setAvailableMemoryProportion({"IPU0": 0.3, "IPU1": 0.3})

# dataset and loader
data_root = './tiles_256_l2/' # input your dataset path
transform = transforms.Compose([
    transforms.ColorJitter(contrast=(0,0.1),saturation=(0,0.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
train_set = CameDataset(data_root=data_root, csv_path=data_root+'train.csv', tile_size=256, transform=transform)
train_loader = poptorch.DataLoader(opts,
                                train_set,
                                batch_size=1,
                                shuffle=True,
                                drop_last=True)

# optimizer
optimizer = poptorch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

print(model)
model.dconv_up2 = poptorch.BeginBlock(model.dconv_up2, ipu_id=1)

model = ModelwithLoss(model, calc_loss)

lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=16, eta_min=1e-7)
training_model = poptorch.trainingModel(model,
                                        options=opts,
                                        optimizer=optimizer)

for epoch in range(1, num_epochs+1):
    epoch_samples = 0
    epoch_loss = 0
    with tqdm(train_loader, unit="batch") as tepoch:

        for inputs, labels in tepoch:

            tepoch.set_description(f"Epoch {epoch+1}")

            output, loss = training_model(inputs, labels)

            # statistics
            epoch_samples += inputs.size(0)
            epoch_loss += loss.mean()
            tepoch.set_postfix(loss=loss.mean(),lr=lr_scheduler.get_last_lr()[0])
            writer.add_scalar("Loss/train", loss.mean(), epoch)
            writer.add_scalar("lr/train", lr_scheduler.get_last_lr()[0], epoch)

            lr_scheduler.step()

        epoch_loss = epoch_loss / (epoch_samples/ipu_batch_size)

    if epoch % 2 == 0 and epoch > 5:
        torch.save(model.state_dict(), f"{epoch}_model.pth")