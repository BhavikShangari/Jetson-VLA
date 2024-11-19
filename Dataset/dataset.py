from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import pandas as pd
import numpy as np
import os

splits = np.linspace(-1,1, 256)
mid = (splits[:-1] + splits[1:]) / 2
base = 127743   ## For Llama Tokenizer

class VLADataset(Dataset):
    def __init__(self, df : str, image_dir:str):
        """
        args:
        df: path of dataframe
        """
        super().__init__()
        self.df = pd.read_json(df)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.image_dir = image_dir
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.df.iloc[idx, 0]))
        prompt= self.df.iloc[idx, 1]
        action = self.df.iloc[idx,2]
        action_tokens = np.digitize(action, splits) + base
        return { 'images': image, 'text': prompt, 'action':action_tokens.tolist()}
        
