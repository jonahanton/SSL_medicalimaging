import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms, datasets
from PIL import Image
import PIL
from tqdm import tqdm

class CustomBachDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False):
        # Random seed
        random_state = 42
        # Read in csv containing path information
        csv_file = os.path.join(img_dir, "ICIAR2018_BACH_Challenge/Photos/microscopy_ground_truth.csv")
        self.preclean_dataframe = pd.read_csv(csv_file,header= None)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        # Split into train and test (80:20)
        split_index = int(math.floor(0.8*len(self.preclean_dataframe)))
        if train: # Train data length: 110
            self.preclean_dataframe = self.preclean_dataframe.iloc[:split_index,:]
        else: # Test data length: 28
            self.preclean_dataframe = self.preclean_dataframe.iloc[split_index:,:]
        # Extract image paths and labels
        self.img_paths, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Extract label
        label_name = self.img_labels.iloc[idx] 
        # Convert to numerical
        if label_name == "Normal":
            label = 0
        elif label_name == "Benign":
            label = 1
        elif label_name == "InSitu":
            label = 2
        elif label_name == "Invasive":
            label = 3
        else:
            raise ValueError
        # Get full image path
        img_path = os.path.join(os.path.join(self.img_dir,"ICIAR2018_BACH_Challenge/Photos/"+label_name),self.img_paths.iloc[idx])
        # Load in RGB image
        image = Image.open(img_path)
        # Convert to appropriate format
        label = np.float32(label)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def _split_labels(self,dataframe):
        """Split Ch dataframe into path, aux and label dataframes"""
        path = dataframe.iloc[:,0]
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

    normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**normalise_dict)

    image_size = 224
    transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    num_classes = 4
    cl_list = range(num_classes)

    sub_meta = {}
    for cl in cl_list:
        sub_meta[cl] = []
    
    trainval_dataset = CustomBachDataset("/vol/bitbucket/g21mscprj03/SSL/data/bach", train=True, transform=transform)
    test_dataset = CustomBachDataset("/vol/bitbucket/g21mscprj03/SSL/data/bach", train=False, transform=transform)
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

def test_class_2():
    cid = CustomBachDataset("/vol/bitbucket/g21mscprj03/SSL/data/bach", train = True)
    print(cid[30])
    print(len(cid))
    cid = CustomBachDataset("/vol/bitbucket/g21mscprj03/SSL/data/bach", train = False)
    print(cid[30])
    print(cid[25])
    print(len(cid))
    print(type(cid[25][1]))

if __name__ == "__main__":
    #test_class_2()
    pass