import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class CustomDiabeticRetinopathyDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False):
        # Random seed
        random_state = 42
        if train:
            self.img_dir = img_dir + "train/"
            csv_file = img_dir + "trainLabels.csv"
        else:
            self.img_dir = img_dir + "test/"
            csv_file = img_dir + "testLabels.csv"
        self.preclean_dataframe = pd.read_csv(csv_file)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        self.img_paths, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        # Need to check whether img_dir has a / at end
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Added extension
        img_path = os.path.join(self.img_dir, self.img_paths.iloc[idx]+".jpeg")
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # image is an Image object, label is a numpy int
        return image, label

    def _clean_labels(self, dataframe):
        # Not sure atm what dataframe looks like, and hence what it should be cleaned with
        return dataframe

    def _split_labels(self,dataframe):
        # Split CheXpert dataframe into path, aux and label dataframes
        partial_path = dataframe.iloc[:,0]
        label = dataframe.iloc[:,1]
        return partial_path, label

    def _basic_preclean(self, dataframe):
        path, label = self._split_labels(dataframe)
        label = self._clean_labels(label)
        return path, label

def test_class():
    # Loads in Correctly (Needs / for img_dir path)
    cid = CustomDiabeticRetinopathyDataset('/vol/bitbucket/lrc121/diabetic_retinopathy/', train = True)
    # For Train:
    # Shuffles Correctly
    # Gives correct img path and label combinations
    # Gives correct length (35126)
    # For Test:
    # Shuffles correctly
    # Gives correct img path and label combinations
    # Gives correct length (53576)
    return None



if __name__ == "__main__":
    pass