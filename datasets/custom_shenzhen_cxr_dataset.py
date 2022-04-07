import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class CustomShenzhenCXRDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False):
        # Random seed
        random_state = 42
        # Note only exists train data!
        csv_file = os.path.join(img_dir, "shenzhen_metadata.csv")
        self.preclean_dataframe = pd.read_csv(csv_file)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        self.img_paths, self.aux, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(os.path.join(self.img_dir, "ChinaSet_AllFiles/CXR_png"), self.img_paths.iloc[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx]
        # print(img_path, label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # image is an Image object, label is a numpy int
        return image, label

    def _clean_labels(self, labels):
        vectorized_convert_to_numerical = np.vectorize(self.convert_to_numerical)
        # 0 is normal 1 is TB
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
        # Split CheXpert dataframe into path, aux and label dataframes
        partial_path = dataframe.iloc[:,0]
        aux = dataframe.iloc[:,1:3]
        label = dataframe.iloc[:,3]
        return partial_path, aux, label

    def _basic_preclean(self, dataframe):
        path, aux, labels = self._split_labels(dataframe)
        labels = self._clean_labels(labels)
        return path, aux, labels

def test_class():
    # Loads in Correctly (Needs / for img_dir path)
    cid = CustomShenzhenCXRDataset('/vol/bitbucket/lrc121/ShenzhenXray/', train = True)
    print(cid[30])
    print(len(cid))
    return None



if __name__ == "__main__":
    # test_class()
    pass