from torchvision.transforms import transforms
from torchvision import datasets
from data.generate_views import GenerateViews


class DatasetGetter:

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.dataset_name = self.args.dataset_name

        self.transforms_database = {
            "MNIST" : GenerateViews(self._return_transforms(28), self.args.n_views),
            "cifar10": GenerateViews(self._return_transforms(32), self.args.n_views),
            }

        self.datasets_database = {
            "MNIST": lambda : datasets.MNIST(self.args.data_path, train=True, download=True,
                                            transform=self.transforms_database["MNIST"]),
            "cifar10": lambda : datasets.CIFAR10(self.args.data_path, train=True, download=True,
                                            transform=self.transforms_database["cifar10"]),
        }

    
    def _return_transforms(self, size):
        
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor()])
        
        return data_transforms
        


    def load(self):

        try:
            data = self.datasets_database[self.dataset_name]
        except KeyError:
            print(f"Dataset {self.dataset_name} not stored in database! Returning None.")
            return

        return data()
        
        