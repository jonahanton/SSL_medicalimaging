import torch
import torch.nn as nn
from torchvision.transforms import transforms
from data.generate_views import GenerateViews

import json

database_path = 'data/datasets_database'

class DatasetGetter:

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        dataset_name = self.args.dataset_name

        with open(f'{database_path}', 'r') as jsonfile:
            database = json.load(jsonfile)


    def load(self):
        pass