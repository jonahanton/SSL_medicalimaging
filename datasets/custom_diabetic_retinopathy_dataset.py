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
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False):
        # Random seed
        random_state = 42
        if train:
            self.img_dir = os.path.join(img_dir, "train")
            csv_file = os.path.join(img_dir, "trainLabels.csv")
        else:
            self.img_dir = os.path.join(img_dir, "test")
            csv_file = os.path.join(img_dir, "testLabels.csv")
        self.preclean_dataframe = pd.read_csv(csv_file)
        #print(self.preclean_dataframe.columns)
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
        label = np.float32(label)
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

    normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**normalise_dict)

    image_size = 224
    transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    # Loads in Correctly (Needs / for img_dir path)
    # cdr = CustomDiabeticRetinopathyDataset('/vol/bitbucket/g21mscprj03/SSL/data/diabetic_retinopathy/', train=True, transform=transform)
    # For Train:
    # Shuffles Correctly
    # Gives correct img path and label combinations
    # Gives correct length (35126)
    # For Test:
    # Shuffles correctly
    # Gives correct img path and label combinations
    # Gives correct length (53576)

    
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
    cid = CustomDiabeticRetinopathyDataset('/vol/bitbucket/g21mscprj03/SSL/data/diabetic_retinopathy', train = True)
    print(cid[30])
    print(len(cid))
    cid = CustomDiabeticRetinopathyDataset('/vol/bitbucket/g21mscprj03/SSL/data/diabetic_retinopathy', train = False)
    print(cid[25])
    print(len(cid))
    print(type(cid[25][1]))
    return None

if __name__ == "__main__":
    #test_class_2()