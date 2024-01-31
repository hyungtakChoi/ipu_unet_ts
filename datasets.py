from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import pandas as pd
import torch
from PIL import Image
import numpy as np

def get_full_path(df, data_root, except_ext=True):
    return f"{data_root}/{df['folder']}/{df['slide']}/{df['file']}"

class CameDataset(Dataset):
    def __init__(self, data_root, csv_path, tile_size=256, transform=None, set_balance=False):

        self.data_df = pd.read_csv(csv_path, index_col=0)

        if set_balance:
            has_no_mask_df = self.data_df.loc[self.data_df.has_mask == False]
            has_mask_df = self.data_df.loc[self.data_df.has_mask == True]
            has_no_mask_df = has_no_mask_df.sample(frac=1)[:len(has_mask_df)]
            self.data_df = pd.concat([has_no_mask_df, has_mask_df])

        self.data_root = data_root

        self.input_images_path = self.data_df.apply(get_full_path, data_root=self.data_root, axis=1).tolist()
        self.has_mask = self.data_df["has_mask"].tolist()
        self.normal_mask = torch.zeros((tile_size, tile_size), dtype=torch.float)

        self.transform = transform
    
    def __len__(self):
        return len(self.input_images_path)
    
    def __getitem__(self, idx):
        if self.has_mask[idx]:
            image = Image.open(self.input_images_path[idx]+'_T.png').convert('RGB')
            mask = Image.open(self.input_images_path[idx].replace("from_ts","from_tsm")+'_T_mask.png')
            #mask = Image.open(self.input_images_path[idx]+'_T_mask.png')
            mask = torch.from_numpy(np.array(mask)).float()
        else:
            image = Image.open(self.input_images_path[idx]+'_N.png').convert('RGB')
            mask = self.normal_mask.clone()
        
        if self.transform:
            image = self.transform(image)

        return (image, mask.unsqueeze(0))

