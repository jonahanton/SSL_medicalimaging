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
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        # Add a bit to split dataframe to train and test (80:20)
        split_index = int(math.floor(0.8*len(self.preclean_dataframe)))
        if train: # Train data length: 1600
            self.preclean_dataframe = self.preclean_dataframe.iloc[:split_index,:]
        else: # Test data length: 400
            self.preclean_dataframe = self.preclean_dataframe.iloc[split_index:,:]
        # Extract the paths and labels from dataframe
        self.img_paths, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load the frame n of the way through the stack
        n = 0.5
        # Find associated label
        label = self.img_labels.iloc[idx] 
        # Converts Label
        label = np.float32(label) 
        # Find full image path
        img_path = os.path.join(self.img_dir,"data/mha/"+str(self.img_paths.iloc[idx])+".mha")
        # Load in mha style image using medpy library
        image, _ = medpy.load(img_path)
        # Extract a slice at a given depth
        depth = math.floor(image.shape[2]*n)
        image = image[:,:,depth]
        # Clip the pixel values
        min_value, max_value = -1000, 250
        np.clip(image, min_value, max_value, out=image)
        # Rescale to [0,1]
        image = (image - min_value) / (max_value - min_value)
        # Rescale to [0,255] and uint8 (as needed for loading L) 
        # Then convert to RGB
        image = Image.fromarray((image* 255).astype(np.uint8)).convert("RGB")
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def _split_labels(self,dataframe):
        """Split STOIC dataframe into path and label dataframes"""
        # Path is the patient ID
        path = dataframe.iloc[:,0]
        # Label is the probability of having COVID (binary)
        label = dataframe.iloc[:,1] 
        return path, label

    def _basic_preclean(self, dataframe):
        """ Converts dataframe to desired format

        Args:
            dataframe (pandas.Dataframe) : original dataframe object with no transformations

        Returns:
            path (pandas.Dataframe) : dataframe object containing path information
            label (pandas.Dataframe) : dataframe object with label information
        """
        path, label = self._split_labels(dataframe)
        return path, label

def test_class():
    cid = CustomStoicDataset("/vol/bitbucket/g21mscprj03/SSL/data/stoic", train = True)
    print(cid[101])
    # print(cid[41])
    # print(cid[42])
    # print(len(cid))
    # cid = CustomStoicDataset("/vol/bitbucket/g21mscprj03/SSL/data/stoic", train = False)
    # print(cid[41])
    # print(len(cid))

if __name__ == "__main__":
    #test_class()
    pass
