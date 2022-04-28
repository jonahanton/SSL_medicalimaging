import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class CustomChexpertDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False, focus = "Pleural Effusion",few_shot = False, group_front_lateral = False):
        # Random seed
        random_state = 42
        # Read in csv containing path information
        csv_file = os.path.join(img_dir, "train.csv")
        self.preclean_dataframe = pd.read_csv(csv_file)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        # Split dataframe to train and test 
        # Validation set is ignored and a custom split is used! 
        # (as in simCLR cheXpert paper)
        if train: # Train data
            self.preclean_dataframe = self.preclean_dataframe.iloc[:134049,:]
        else: # Test Data
            self.preclean_dataframe = self.preclean_dataframe.iloc[134049:,:]
        # Pick only the laterals (for multiview grouping)
        self.group_front_lateral = group_front_lateral
        if self.group_front_lateral:
            self.preclean_dataframe = self.find_only_laterals(self.preclean_dataframe)
        self.few_shot = few_shot
        # Extract the paths and labels from dataframe
        self.img_paths, self.img_aux, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # Specify label for many to one
        try:
            self.many_to_one_label = ["Atelectasis", "Cardiomegaly", "Consolidation","Edema", "Pleural Effusion"].index(focus)
        except: # If above fails, is because an invalid option is given!
            raise ValueError(f"{focus} is not one of the allowed many_to_one labels (Atelectasis, Cardiomegaly, Consolidation,Edema, Pleural Effusion)")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Create full image path
        img_path = os.path.join(self.img_dir, self.img_paths.iloc[idx])
        # Open image with PIL, stack grayscale images to create 3 channel
        image = Image.open(img_path).convert('RGB')
        # Extract the full label of image
        multi_label = self.img_labels.iloc[idx]
        # Convert to many to one label
        label = multi_label.to_numpy()[self.many_to_one_label]
        # Ensure label is in correct format
        label = np.float32(label)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.group_front_lateral:
            # Find frontal image associated with lateral image given
            image2 = self.group_additional_images(img_path)
            image = (image,image2)
        return image, label
    
    def group_additional_images(self, img_path):
        """ Find the frontal image associated with given lateral image

        Args:
            img_path (str) : full path to lateral image

        Returns:
            PIL.Image : a PIL Image object of the frontal image
        """
        # Convert to list
        path_list = img_path.split("/")
        # Replace final bit of path with frontal image
        path_list[-1] = "view1_frontal.jpg"
        img_path = "/".join(path_list)
        # open Image object
        image = Image.open(img_path).convert('RGB')
        # apply any given transformations
        if self.transform:
            image = self.transform(image)
        return image


    def _clean_labels(self, dataframe):
        """ Converts labels to desired format

        Args:
            dataframe (pandas.Dataframe) : dataframe object with unconverted labels

        Returns:
            dataframe (pandas.Dataframe) : dataframe object with converted labels
        """
        # Fill NaNs with 0s
        filled_dataframe = dataframe.fillna(0.0)
        # few shot needs binary labels
        if self.few_shot:
            # Transform uncertainties to positive cases
            vectorized_u_fn = np.vectorize(self._u_ones)
        else:
            # Transform uncertainties to positive cases (using label smoothing)
            vectorized_u_fn = np.vectorize(self._u_ones_lsr)
        # Apply vectorized function to dataframe
        smoothed_dataframe = filled_dataframe.transform(vectorized_u_fn)
        return smoothed_dataframe

    def _u_ones_lsr(self, val):
        """Convert uncertainty value to a smoothed positive label"""
        # From Pham et al. (pg 13)
        low = 0.55
        high = 0.85
        # Convert positive and negative values to float
        if math.isclose(val, 1.0) or math.isclose(val, 0.0):
            return float(val)
        # Convert uncertainty values to random uniform
        else:
            return np.random.uniform(low, high)
    
    def _u_ones(self, val):
        """Convert uncertainty value to a positive label"""
        # From Pham et al. (pg 13)
        # Convert positive and negative values to float
        if math.isclose(val, 1.0) or math.isclose(val, 0.0):
            return float(val)
        # Convert uncertainty values to 1
        else:
            return 1.0
    
    def find_only_laterals(self,dataframe):
        """Extract only the image paths which are lateral images"""
        dataframe = dataframe.loc[dataframe["Frontal/Lateral"] == "Lateral"]
        return dataframe

    def _split_labels(self,dataframe):
        """Split dataframe into path, aux and label dataframes"""
        path = dataframe.iloc[:,0]
        aux = dataframe.iloc[:,1:5]
        label = dataframe.iloc[:,6:]
        # Only take the 5 pathologies as specified in Azizi et al.
        label = label[["Atelectasis", "Cardiomegaly", "Consolidation","Edema", "Pleural Effusion"]]
        return path, aux, label

    def _basic_preclean(self, dataframe):
        """ Converts dataframe to desired format

        Args:
            dataframe (pandas.Dataframe) : original dataframe object with no transformations

        Returns:
            path (pandas.Dataframe) : dataframe object containing path information
            aux (pandas.Dataframe) : dataframe object with auxiliary information
            label (pandas.Dataframe) : dataframe object with label information
        """
        path, aux, label = self._split_labels(dataframe)
        label = self._clean_labels(label)
        return path, aux, label

def test_class():
    cid = CustomChexpertDataset("/vol/bitbucket/g21mscprj03/SSL/data/chexpert", train = True, few_shot = True, group_front_lateral = True)
    cid = CustomChexpertDataset("/vol/bitbucket/g21mscprj03/SSL/data/chexpert", train = False, few_shot = True, group_front_lateral = True)


if __name__ == "__main__":
    #test_class()
    pass