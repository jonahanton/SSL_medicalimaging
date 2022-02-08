from torchvision.transforms import transforms
from torchvision import datasets
from data.generate_views import GenerateViews


class DatasetGetter:

    def __init__(self, *args, pretrain=True, train=True, **kwargs):
        self.args = kwargs['args']
        self.dataset_name = self.args.dataset_name

        # Is this dataset being loading in for ssl pre training?
        self.pretrain = pretrain
        # Do we want to load the train or the test dataset?
        self.train = train

        self.transforms_database = {
            "MNIST" : GenerateViews(self._return_transforms(28), self.args.n_views),
            "cifar10": GenerateViews(self._return_transforms(32), self.args.n_views),
            "CheXpert_small": GenerateViews(self._return_transforms(128), self.args.n_views) # look at paper
            }

        # Datasets that are available only on DOC Bitbucket
        CheXpert_small_data_path = "~/vol/bitbucket/lrc121/CheXpert_data_small/CheXpert-v1.0-small/train" # need to check if can load in properly

        if self.pretrain:
                self.datasets_database = {
                "MNIST": lambda : datasets.MNIST(self.args.data_path, train=self.train, download=True,
                                                transform=self.transforms_database["MNIST"]),
                "cifar10": lambda : datasets.CIFAR10(self.args.data_path, train=self.train, download=True,
                                                transform=self.transforms_database["cifar10"]),
                "CheXpert_small": lambda: datasets.ImageFolder(CheXpert_small_data_path,
                                                transform=self.transforms_database["CheXpert_small"]),
            }
        else:
            self.datasets_database = {
                "MNIST": lambda : datasets.MNIST(self.args.data_path, train=self.train, download=True,
                                                transform=self._return_transforms(28)),
                "cifar10": lambda : datasets.CIFAR10(self.args.data_path, train=self.train, download=True,
                                                transform=self._return_transforms(32)),
                "CheXpert_small": lambda: datasets.ImageFolder(CheXpert_small_data_path, train=self.train, download=False,
                                                    transform=self._return_transforms(128)),
            }

    
    def _return_transforms(self, size):
        
        if self.pretrain:
            # Data augmentations to apply for pretraining
            color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor()])
        else:
            # Data augmentations to apply to dataset when loading in for fine-tuning/linear classifier
            # Not sure which to apply here? - Just transforms.ToTensor()? What about .Normalize()? Also assume this is dataset specific...?
            data_transforms = transforms.Compose([transforms.ToTensor()])
                                            
        return data_transforms
        

    def load(self):

        try:
            data = self.datasets_database[self.dataset_name]
        except KeyError:
            print(f"Dataset {self.dataset_name} not stored in database! Returning None.")
            return

        return data()
        
        