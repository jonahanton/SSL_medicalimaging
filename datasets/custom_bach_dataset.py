import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class CustomBachDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False):
        # Random seed
        random_state = 42
        # Read in csv containing path information
        csv_file = os.path.join(img_dir, "ICIAR2018_BACH_Challenge/Photos/microscopy_ground_truth.csv")
        self.preclean_dataframe = pd.read_csv(csv_file,header= None)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        # Add a bit to split dataframe to train and test

        self.img_paths, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx] # Is a numpy array!
        if np.array_equal(label, np.array([1,0,0,0])):
            label_name = "Normal"
        elif np.array_equal(label, np.array([0,1,0,0])):
            label_name = "Benign"
        elif np.array_equal(label, np.array([0,0,1,0])):
            label_name = "InSitu"
        elif np.array_equal(label, np.array([0,0,0,1])):
            label_name = "Invasive"
        else:
            raise ValueError
        img_path = os.path.join(os.path.join(self.img_dir,"ICIAR2018_BACH_Challenge/Photos/"+label_name),self.img_paths.iloc[idx])
        image = Image.open(img_path) #RGB
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def _clean_labels(self, dataframe):
        vectorized_convert_to_numerical = np.vectorize(self.convert_to_numerical)
        smoothed_dataframe = dataframe.transform(vectorized_convert_to_numerical)
        return smoothed_dataframe
    
    def convert_to_numerical(self,val):
        if val == "Normal":
            val = [1,0,0,0]
        elif val == "Benign":
            val = [0,1,0,0]
        elif val == "InSitu":
            val = [0,0,1,0]
        elif val == "Invasive":
            val = [0,0,0,1]
        else:
            raise ValueError
        return val


    def _split_labels(self,dataframe):
        # Split CheXpert dataframe into path, aux and label dataframes
        path = dataframe.iloc[:,0]
        label = dataframe.iloc[:,1]
        return path, label

    def _basic_preclean(self, dataframe):
        path, label = self._split_labels(dataframe)
        label = self._clean_labels(label)
        return path, label

def test_class():
    cid = CustomBachDataset("/vol/bitbucket/g21mscprj03/SSL/data/bach", train = True)


if __name__ == "__main__":
    pass