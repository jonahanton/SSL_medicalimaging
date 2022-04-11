import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class CustomiChallengePMDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False):
        # Random seed
        random_state = 42
        # Read in csv containing path information
        csv_file = os.path.join(img_dir, "PALM-Training400/fovea_location_csv.csv")
        self.preclean_dataframe = pd.read_csv(csv_file)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        # Add a bit to split dataframe to train and test (80:20)
        split_index = int(math.floor(0.8*len(self.preclean_dataframe)))
        if train: # Train data
            self.preclean_dataframe = self.preclean_dataframe.iloc[:split_index,:]
        else: # Test Data
            self.preclean_dataframe = self.preclean_dataframe.iloc[split_index:,:]
        self.img_paths, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx] 
        label = np.float32(label) # Converts Label
        # Combine labels for classes 1 and 2 together 
        if label == 2:
            label = np.float32(1)
        img_path = os.path.join(os.path.join(self.img_dir,"PALM-Training400/PALM-Training400/"),self.img_paths.iloc[idx])
        # print(img_path, label)
        image = Image.open(img_path) # RGB
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def _clean_labels(self, dataframe):
        return dataframe
    
    def _split_labels(self,dataframe):
        # Split CheXpert dataframe into path, aux and label dataframes
        path = dataframe.iloc[:,1]
        label = dataframe.iloc[:,4]
        return path, label

    def _basic_preclean(self, dataframe):
        path, label = self._split_labels(dataframe)
        label = self._clean_labels(label)
        return path, label

def test_class():
    cid = CustomiChallengePMDataset("/vol/bitbucket/g21mscprj03/SSL/data/ichallenge_pm", train = True)
    print(cid[10])
    print(len(cid))
    cid = CustomiChallengePMDataset("/vol/bitbucket/g21mscprj03/SSL/data/ichallenge_pm", train = False)
    print(cid[10])
    print(len(cid))
if __name__ == "__main__":
    #test_class()
    pass