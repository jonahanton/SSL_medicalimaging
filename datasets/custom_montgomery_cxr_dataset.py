import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class CustomMontgomeryCXRDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False):
        # Random seed
        random_state = 42
        # Load in csv file
        img_path = os.path.join(img_dir, "montgomery_metadata.csv")
        self.preclean_dataframe = pd.read_csv(img_path)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        # Split into train and test (80:20)
        split_index = int(math.floor(0.8*len(self.preclean_dataframe)))
        if train: # Train data length: 110
            self.preclean_dataframe = self.preclean_dataframe.iloc[:split_index,:]
        else: # Test data length: 28
            self.preclean_dataframe = self.preclean_dataframe.iloc[split_index:,:]
        # Extract image paths and labels
        self.img_paths, self.aux, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load in image
        img_path = os.path.join(os.path.join(self.img_dir, "MontgomerySet/CXR_png/"), self.img_paths.iloc[idx])
        image = Image.open(img_path).convert('RGB')
        # Get label in correct format
        label = self.img_labels.iloc[idx]
        label = np.float32(label)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def _clean_labels(self, labels):
        """Convert categorical labels into numeric"""
        vectorized_convert_to_numerical = np.vectorize(self.convert_to_numerical)
        numerical_labels = labels.transform(vectorized_convert_to_numerical)
        return numerical_labels
    
    def convert_to_numerical(self, val):
        # We assume that anything not normal is TB
        if val == "normal":
            val = 0
        else:
            val = 1
        return val

    def _split_labels(self,dataframe):
        """Split dataframe into path, aux and label dataframes"""
        partial_path = dataframe.iloc[:,0]
        aux = dataframe.iloc[:,1:3]
        label = dataframe.iloc[:,3]
        return partial_path, aux, label

    def _basic_preclean(self, dataframe):
        """ Converts dataframe to desired format

        Args:
            dataframe (pandas.Dataframe) : original dataframe object with no transformations

        Returns:
            path (pandas.Dataframe) : dataframe object containing path information
            aux (pandas.Dataframe) : dataframe object with auxiliary information
            label (pandas.Dataframe) : dataframe object with label information
        """
        path, aux, labels = self._split_labels(dataframe)
        labels = self._clean_labels(labels)
        return path, aux, labels

def test_class():
    # Loads in Correctly (Needs / for img_dir path)
    cid = CustomMontgomeryCXRDataset('/vol/bitbucket/g21mscprj03/SSL/data/montgomerycxr', train = True)
    print(cid[30])
    print(len(cid))
    cid = CustomMontgomeryCXRDataset('/vol/bitbucket/g21mscprj03/SSL/data/montgomerycxr', train = False)
    print(cid[25])
    print(len(cid))
    print(type(cid[25][1]))
    return None



if __name__ == "__main__":
    #test_class()
    pass