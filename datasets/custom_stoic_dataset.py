import os
import pandas as pd
import numpy as np
import math
import medpy.io as medpy
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class CustomStoicDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False):
        # Random seed
        random_state = 42
        # Read in csv containing path information
        csv_file = os.path.join(img_dir, "metadata/reference.csv")
        self.preclean_dataframe = pd.read_csv(csv_file)

        print(self.preclean_dataframe.columns)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        
        # Add a bit to split dataframe to train and test (80:20)
        split_index = int(math.floor(0.8*len(self.preclean_dataframe)))
        if train: # Train data (1600)
            self.preclean_dataframe = self.preclean_dataframe.iloc[:split_index,:]
        else: # Test Data (400)
            self.preclean_dataframe = self.preclean_dataframe.iloc[split_index:,:]
        self.img_paths, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels*375)

    def __getitem__(self, idx):
        # Load the nth frame
        n = 5
        label = self.img_labels.iloc[idx] 
        label = np.float32(label) # Converts Label
        img_path = os.path.join(self.img_dir,"data/mha/"+str(self.img_paths.iloc[idx])+".mha")
        print(img_path)
        image, image_header = medpy.load(img_path)
        image = image[:,:,n]
        # # Not including atm as clipping was not done in all CT scan images: https://github.com/UCSD-AI4H/COVID-CT/blob/master/baseline%20methods/Self-Trans/CT-predict-pretrain.ipynb
        # # With clippings as in https://github.com/bkong999/COVNet/blob/master/dataset.py
        # print(np.max(image), np.min(image)) # 1586 -2048 (but have also seen 1246 -2048)
        # min_value, max_value = -1250, 250
        # np.clip(image, min_value, max_value, out=image)
        # image = (image - min_value) / (max_value - min_value)
        image = Image.fromarray(image).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def _clean_labels(self, dataframe):
        return dataframe
    
    def _split_labels(self,dataframe):
        # Split STOIC dataframe into path and label dataframes
        path = dataframe.iloc[:,0] # Patient ID
        label = dataframe.iloc[:,1] # probCOVID
        # aux = dataframe.iloc[:,2] # probSevere
        return path, label

    def _basic_preclean(self, dataframe):
        path, label = self._split_labels(dataframe)
        label = self._clean_labels(label)
        return path, label

def test_class():
    cid = CustomStoicDataset("/vol/bitbucket/g21mscprj03/SSL/data/stoic", train = True)
    print(cid[40])
    print(cid[41])
    print(cid[42])
    print(len(cid))
    cid = CustomStoicDataset("/vol/bitbucket/g21mscprj03/SSL/data/stoic", train = False)
    print(cid[41])
    print(len(cid))
if __name__ == "__main__":
    #test_class()
    pass