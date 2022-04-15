import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image

class CustomChestXDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False):
        """
        Args:
            root (string): path to dataset
        """
        self.root = img_dir
        self.img_path = self.root + "/images/"
        self.csv_path = self.root + "/Data_Entry_2017.csv"
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}

        labels_set = []

        # Transforms
        self.transform = transform
        self.target_transform = target_transform
        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)
        # Shuffle
        random_state = 42
        self.data_info = self.data_info.sample(frac=1, random_state = random_state).reset_index(drop=True)
        split_index = int(math.floor(0.8*len(self.data_info)))
        if train: # Train data
            self.data_info = self.data_info.iloc[:split_index,:]
        else: # Test Data
            self.data_info = self.data_info.iloc[split_index:,:]
        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.labels = []


        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)
    
        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)        

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        # Open image
        img_path = os.path.join(self.img_path, single_image_name)
        img = Image.open(img_path).convert('RGB')

        # Transform
        if self.transform:
            img = self.transform(img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index] # int64
        single_image_label = np.float32(single_image_label) #convert
        if self.target_transform:
            single_image_label = self.target_transform(single_image_label)
        
        return img, single_image_label

    def __len__(self):
        return self.data_len

def test_class():
    cid = CustomChestXDataset("/vol/bitbucket/g21mscprj03/SSL/data/chestx", train = True)
    print(cid[40])
    print(cid[41])
    print(cid[42])
    print(cid[43])
    print(cid[44])
    print(type(cid[42][1]))
    print(len(cid))
    cid = CustomChestXDataset("/vol/bitbucket/g21mscprj03/SSL/data/chestx", train = False)
    print(cid[40])
    print(cid[41])
    print(cid[42])
    print(cid[43])
    print(cid[44])
    print(len(cid))
if __name__ == "__main__":
    #test_class()
    pass