import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms, datasets
import PIL
from PIL import Image
import pickle

class CustomDiabeticRetinopathyDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False, group_front_lateral = False):
        # Random seed
        random_state = 42
        # Load the appropriate csv file
        if train:
            self.img_dir = os.path.join(img_dir, "train")
            csv_file = os.path.join(img_dir, "trainLabels.csv")
        else:
            self.img_dir = os.path.join(img_dir, "test")
            csv_file = os.path.join(img_dir, "testLabels.csv")
        # Read in csv file
        self.preclean_dataframe = pd.read_csv(csv_file)
        # Group left and right eye images (Note that we keep the 
        # same keyword argument to be consistent with chexpert dataset loader)
        self.group_left_right = group_front_lateral
        if self.group_left_right:
            self.preclean_dataframe = self.return_only_right_views(self.preclean_dataframe)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        # Extract image paths and labels
        self.img_paths, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get full path
        img_path = os.path.join(self.img_dir, self.img_paths.iloc[idx]+".jpeg")
        # Open RGB image
        image = Image.open(img_path)
        # Extract label
        label = self.img_labels.iloc[idx]
        # Convert to correct format
        label = np.float32(label)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # Get multiview images
        if self.group_left_right:
            image2 = self.group_additional_images(img_path)
            image = (image,image2)
        return image, label
    
    def group_additional_images(self, img_path):
        """ Find the right eye image associated with given left eye image

        Args:
            img_path (str) : full path to left eye image

        Returns:
            PIL.Image : a PIL Image object of the right eye image
        """
        path_list = img_path.split("/")
        # Get image path of right eye image
        number = path_list[-1][:-10]
        path_list[-1] = number+"_right.jpeg"
        img_path = "/".join(path_list)
        # Open image
        image = Image.open(img_path)
        # Apply transformation
        if self.transform:
            image = self.transform(image)
        return image

    def return_only_right_views(self,dataframe):
        """Get images where both eyes have same label"""
        # Get only the right eyes
        dataframe_left = dataframe.iloc[::2,:].reset_index(drop=True)
        # Get only the left eyes
        dataframe_right = dataframe.iloc[1::2,:].reset_index(drop=True)
        # Create new column with bool if eyes have same labels
        dataframe_left['bool_results'] = np.where(dataframe_left["level"] == dataframe_right["level"], True, False)
        # Extract only those where labels match
        dataframe = dataframe_left[dataframe_left["bool_results"] == True].reset_index(drop=True)
        return dataframe

    def _split_labels(self,dataframe):
        """Split dataframe into path, aux and label dataframes"""
        partial_path = dataframe.iloc[:,0]
        label = dataframe.iloc[:,1]
        return partial_path, label

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

    normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**normalise_dict)

    image_size = 224
    transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    num_classes = 5
    cl_list = range(num_classes)

    sub_meta = {}
    for cl in cl_list:
        sub_meta[cl] = []
    
    trainval_dataset = CustomDiabeticRetinopathyDataset('/vol/bitbucket/g21mscprj03/SSL/data/diabetic_retinopathy/', train=True, transform=transform)
    test_dataset = CustomDiabeticRetinopathyDataset('/vol/bitbucket/g21mscprj03/SSL/data/diabetic_retinopathy/', train=False, transform=transform)
    d = ConcatDataset([trainval_dataset, test_dataset])

    print(f'Total dataset size: {len(d)}')

    pbar = tqdm(range(len(d)), desc='Iterating through dataset')
    for i, (data, label) in enumerate(d):
        sub_meta[label].append(data)
        pbar.update(1)
    pbar.close()
    
    print('Number of images per class')
    for key, item in sub_meta.items():
        print(len(sub_meta[key]))

    
    with open('sub_meta_diabetic_retinopathy.pickle', 'wb') as handle:
        pickle.dump(sub_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test_class_2():
    # Loads in Correctly (Needs / for img_dir path)
    cid = CustomDiabeticRetinopathyDataset('/vol/bitbucket/g21mscprj03/SSL/data/diabetic_retinopathy', train = True, group_left_right=True)
    print(cid[15])
    cid = CustomDiabeticRetinopathyDataset('/vol/bitbucket/g21mscprj03/SSL/data/diabetic_retinopathy', train = False, group_left_right=True)
    print(cid[15])
    return None

if __name__ == "__main__":
    #test_class_2()
    pass