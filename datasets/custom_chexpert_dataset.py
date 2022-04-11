import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class CustomChexpertDataset(Dataset):
    def __init__(self, img_dir, train = False, transform=None, target_transform=None, download=False, focus = "Pleural Effusion"):
        # Random seed
        random_state = 42
        # Read in csv containing path information
        csv_file = os.path.join(img_dir, "train.csv")
        self.preclean_dataframe = pd.read_csv(csv_file)
        # Shuffle dataframe
        self.preclean_dataframe = self.preclean_dataframe.sample(frac=1, random_state = random_state).reset_index(drop=True)
        # Add a bit to split dataframe to train and test
        # Validation set is ignored! (as in simCLR cheXpert paper)
        if train: # Train data
            self.preclean_dataframe = self.preclean_dataframe.iloc[:134049,:]
        else: # Test Data
            self.preclean_dataframe = self.preclean_dataframe.iloc[134049:,:]
        self.img_paths, self.img_aux, self.img_labels = self._basic_preclean(self.preclean_dataframe) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        try:
            self.many_to_one_label = ["Atelectasis", "Cardiomegaly", "Consolidation","Edema", "Pleural Effusion"].index(focus)
        except:
            raise ValueError(f"{focus} is not one of the allowed many_to_one labels (Atelectasis, Cardiomegaly, Consolidation,Edema, Pleural Effusion)")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths.iloc[idx])
        #print(img_path)
        image = Image.open(img_path).convert('RGB')
        multi_label = self.img_labels.iloc[idx]
        label = multi_label.to_numpy()[self.many_to_one_label]
        label = np.float32(label)
        #print(label, multi_label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def _clean_labels(self, dataframe):
        # May need to adjust this depending on whether is few shot or not

        # Fill NaNs with 0s (nothing noted anywhere about it)
        filled_dataframe = dataframe.fillna(0.0)
        # Transform uncertainties (using label smoothing)
        # Note: The best performer on CheXpert (Yuan et al. 2021 Deep AUC) used the label
        # smoothing technique from (Pham et al. Interpreting chest 
        # x-rays via cnns that exploit hierarchical disease dependencies 
        # and uncertainty labels.)
        vectorized_u_ones_lsr = np.vectorize(self._u_ones_lsr)
        smoothed_dataframe = filled_dataframe.transform(vectorized_u_ones_lsr)
        return smoothed_dataframe

    def _u_ones_lsr(self, val):
        # From Pham et al. (pg 13)
        low = 0.55
        high = 0.85
        if math.isclose(val, 1.0) or math.isclose(val, 0.0):
            return float(val)
        else:
            return np.random.uniform(low, high)

    def _split_labels(self,dataframe):
        # Split CheXpert dataframe into path, aux and label dataframes
        path = dataframe.iloc[:,0]
        aux = dataframe.iloc[:,1:5]
        label = dataframe.iloc[:,6:]
        # Only take the 5 pathologies as specified in Azizi et al.
        label = label[["Atelectasis", "Cardiomegaly", "Consolidation","Edema", "Pleural Effusion"]]
        return path, aux, label

    def _basic_preclean(self, dataframe):
        path, aux, label = self._split_labels(dataframe)
        label = self._clean_labels(label)
        return path, aux, label

def test_class():
    cid = CustomChexpertDataset("/vol/bitbucket/g21mscprj03/SSL/data/chexpert", train = True)
    print(cid[5000])
    print(type(cid[5000][1]))
    print(len(cid))
    cid = CustomChexpertDataset("/vol/bitbucket/g21mscprj03/SSL/data/chexpert", train = False)
    print(cid[5000])
    print(len(cid))


if __name__ == "__main__":
    #test_class()
    pass